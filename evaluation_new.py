#!/usr/bin/env python3
"""
=================================================================
Usage:
    python evaluation_new.py --base_dir inspection_outputs --fasta_file h3n2_clean.fasta --preds_dir inspection_outputs --vaccine_dir Vaccine_strains

Description:
    Evaluates saved FASTA and CSV prediction files against ground truth.
    Features: 
      - Population-weighted positional accuracy
      - BLOSUM62 Distances for: Shifted sites, All sites, Epitope sites
      - Single prediction per model per season, evaluated against ALL test seqs.
      - Maps CSV predictions from the introspection output directory automatically.
      - Automatically links vaccine FASTAs by year.
      - Detailed logs of seasons, vaccine FASTAs, and prediction CSVs
=================================================================
"""

import os
import re
import glob
import argparse
import numpy as np
from collections import Counter
import pandas as pd
from Bio import SeqIO
from Bio.Align import substitution_matrices
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

try:
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
except Exception as e:
    print(f"[ERROR] Could not load BLOSUM62: {e}")
    exit(1)

# ── Epitope Definitions ────────────────────────────────────────────────────────
EPITOPE_SITES_1_BASED = {
    'A': [122,124,126,130,131,132,133,135,137,138,140,142,143,144,145,146,150,152,168],
    'B': [128,129,155,156,157,158,159,160,163,164,165,186,187,188,189,190,192,193,194,196,197,198],
    'C': [44,45,46,47,48,50,51,53,54,273,275,276,278,279,280,294,297,299,300,304,305,307,308,309,310,311,312],
    'D': [96,102,103,117,121,167,170,171,172,173,174,175,176,177,179,182,201,203,207,208,209,
          212,213,214,215,216,217,218,219,226,227,228,229,230,238,240,242,244,246,247,248],
    'E': [57,59,62,63,67,75,78,80,81,82,83,86,87,88,91,92,94,109,260,261,262,265],
}
EPITOPE_0_BASED = {pos - 1 for sites in EPITOPE_SITES_1_BASED.values() for pos in sites}

def load_fasta(fasta_path, zone_idx):
    print(f"Loading Master FASTA: {fasta_path}...")
    all_parsed_data = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        fields = record.id.split('|')
        parsed_date = None
        
        for field in fields:
            try:
                parsed_date = pd.to_datetime(field.strip(), format='%Y-%m-%d')
                break 
            except ValueError:
                continue
                
        if parsed_date is None: continue
            
        zone = "Global"
        if len(fields) > zone_idx:
            zone = fields[zone_idx].strip()
            
        sequence = str(record.seq).replace('-', '').replace('*', '')
        all_parsed_data.append({'id': record.id, 'date': parsed_date, 'zone': zone, 'sequence': sequence})
        
    df = pd.DataFrame(all_parsed_data)
    print(f"✅ Loaded {len(df)} valid sequences.")
    return df

def get_consensus(seq_list, seq_max_len):
    consensus = []
    for pos in range(seq_max_len):
        aas = [s[pos] for s in seq_list if pos < len(s) and s[pos] not in ['-', 'X', '<PAD>']]
        consensus.append(str(Counter(aas).most_common(1)[0][0]).strip().upper() if aas else '-')
    return consensus

def get_positional_frequencies(seq_list, seq_max_len):
    pos_counts = {}
    for pos in range(seq_max_len):
        aas = [s[pos].upper() for s in seq_list if pos < len(s) and s[pos] not in ['-', 'X', '<PAD>']]
        pos_counts[pos] = Counter(aas)
    return pos_counts

def get_aa_distribution(pos_counts, pos, total_seqs):
    if pos not in pos_counts or total_seqs == 0: return "None (0.00)"
    return ", ".join([f"{aa} ({count/total_seqs:.2f})" for aa, count in pos_counts[pos].most_common()])

def parse_predicted_fasta(filepath):
    try:
        with open(filepath, 'r') as f: 
            return "".join([line.strip() for line in f if not line.startswith('>')]).replace('-', '').replace('*', '').strip()
    except: return None

def calc_blosum_distance(pred_aa, target_aa):
    if pred_aa == target_aa: return 0
    if pred_aa not in BLOSUM62.alphabet or target_aa not in BLOSUM62.alphabet: return 10 
    try:
        score_pred = BLOSUM62[pred_aa, pred_aa]
        score_target = BLOSUM62[target_aa, target_aa]
        sub_score = BLOSUM62.get((pred_aa, target_aa), BLOSUM62.get((target_aa, pred_aa), -4))
        return score_pred + score_target - (2 * sub_score)
    except: return 10

def get_sequence_distances(pred_seq, test_seqs, positions_to_eval):
    distances = []
    for t_seq in test_seqs:
        dist = sum([calc_blosum_distance(pred_seq[p].upper() if p < len(pred_seq) else '-', t_seq[p].upper() if p < len(t_seq) else '-') for p in positions_to_eval])
        distances.append(dist)
    return distances

def get_sequence_accuracies(pred_seq, test_seqs, positions_to_eval):
    if not positions_to_eval: 
        return [0.0] * len(test_seqs)
    
    accuracies = []
    total_pos = len(positions_to_eval)
    for t_seq in test_seqs:
        matches = sum(1 for p in positions_to_eval if 
                      (pred_seq[p].upper() if p < len(pred_seq) else '-') == 
                      (t_seq[p].upper() if p < len(t_seq) else '-'))
        accuracies.append(matches / total_pos)
    return accuracies

def get_stats(dist_list):
    if dist_list: return np.min(dist_list), np.median(dist_list), np.mean(dist_list), np.max(dist_list)
    return 0, 0, 0, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Master backtest folder")
    parser.add_argument("--fasta_file", type=str, required=True, help="Master FASTA file")
    parser.add_argument("--zone_idx", type=int, default=1, help="Index of the Zone in header")
    parser.add_argument("--preds_dir", type=str, default="./introspection_out", help="Folder where CSV predictions are saved")
    parser.add_argument("--vaccine_dir", type=str, default="Vaccine_strains", help="Directory containing seasonal vaccine strain FASTA files")
    args = parser.parse_args()

    out_dir = os.path.join(args.base_dir, f"{os.path.splitext(os.path.basename(args.fasta_file))[0]}_evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    master_df = load_fasta(args.fasta_file, args.zone_idx)
    seq_max_len = int(master_df['sequence'].str.len().max())
    
    # ── 1. Pre-load Vaccines and Print Count ──────────────────────────────────
    vaccine_dict = {}
    vaccine_dir = args.vaccine_dir
    
    print("\n" + "═"*70)
    print("💉 VACCINE STRAINS SCAN")
    print("═"*70)
    if os.path.exists(vaccine_dir):
        vaccine_patterns = [
            os.path.join(vaccine_dir, "**", "*.fasta"),
            os.path.join(vaccine_dir, "**", "*.fa"),
            os.path.join(vaccine_dir, "**", "*.fna")
        ]
        vaccine_files = []
        for pat in vaccine_patterns:
            vaccine_files.extend(glob.glob(pat, recursive=True))

        for fpath in sorted(vaccine_files):
            match = re.search(r'(20\d{2})', os.path.basename(fpath).lower())
            if match: 
                year = int(match.group(1))
                vaccine_dict[year] = parse_predicted_fasta(fpath)
                print(f"  -> [Year {year}] Loaded: {os.path.basename(fpath)}")
        print(f"✅ Found and loaded {len(vaccine_dict)} vaccine strain FASTA file(s) from '{vaccine_dir}'.")
    else:
        print(f"⚠️ Notice: Vaccine folder '{vaccine_dir}' not found. Skipping vaccines.")

    # ── 2. Discover Target Seasons and Print Count ────────────────────────────
    target_years = set()
    if os.path.exists(args.base_dir):
        for f in os.listdir(args.base_dir):
            if "season" in f:
                match = re.search(r"(\d{4})", f)
                if match: target_years.add(int(match.group(1)))
            
    if os.path.exists(args.preds_dir):
        for f in os.listdir(args.preds_dir):
            match = re.search(r"(\d{4})", f)
            if match: target_years.add(int(match.group(1)))
            
    target_years = sorted(list(target_years))
    
    print("\n" + "═"*70)
    print("📁 TARGET SEASONS / FOLDERS DISCOVERY")
    print("═"*70)
    print(f"✅ Total Seasons / Folders Found: {len(target_years)}")
    print(f"   Seasons: {target_years}\n")

    if not target_years:
        print("❌ No target seasons found! Exiting.")
        return

    all_summary_results, all_shift_motifs, all_shifted_positions_detailed = [], [], []
    all_position_predictions, all_raw_distances, all_raw_accuracies = [], [], []
    season_sequence_counts, evaluated_test_sequences = [], []

    # ── 3. Loop Through Seasons and Inspect Predictions ───────────────────────
    for target_year in target_years:
        print("\n" + "─"*70)
        print(f"🚀 SEASON: {target_year}")
        print("─"*70)
        
        ref_start, ref_end = pd.to_datetime(f"{target_year - 1}-07-01"), pd.to_datetime(f"{target_year}-01-31")
        test_start, test_end = pd.to_datetime(f"{target_year}-07-01"), pd.to_datetime(f"{target_year + 1}-01-31")

        season_ref_df = master_df[(master_df['date'] >= ref_start) & (master_df['date'] <= ref_end)]
        season_test_df = master_df[(master_df['date'] >= test_start) & (master_df['date'] <= test_end)]
        
        if season_ref_df.empty or season_test_df.empty: 
            print(f"  ⚠️ Skipping Season {target_year}: Not enough sequences (Ref: {len(season_ref_df)}, Test: {len(season_test_df)}).")
            continue
            
        season_preds = {} 
        
        # 3a. Search and Print Prediction CSVs for this season
        pred_year_dir = os.path.join(args.preds_dir, str(target_year))
        csv_files = []
        if os.path.exists(pred_year_dir):
            all_csvs = glob.glob(os.path.join(pred_year_dir, "**", "*.csv"), recursive=True)
            csv_files = [c for c in all_csvs if not any(x in os.path.basename(c) for x in ["Summary", "Distance", "Motifs", "Counts", "Evaluated", "Detailed", "Accuracies"])]
            
            print(f"  📊 Prediction CSVs Found for {target_year}: {len(csv_files)} file(s)")
            
            for csv_path in csv_files:
                fname = os.path.basename(csv_path)
                try:
                    df_csv = pd.read_csv(csv_path)
                    model_name = os.path.basename(os.path.dirname(csv_path))
                    
                    if 'Predicted_Sequence' in df_csv.columns:
                        # Grab the single predicted sequence for this model/season
                        predicted_seq = str(df_csv['Predicted_Sequence'].dropna().iloc[0]).strip()
                        season_preds[model_name] = predicted_seq
                        print(f"     └─ Model '{model_name}': Loaded prediction from {fname}")
                    else:
                        print(f"     └─ ⚠️ Skipped {fname}: Missing 'Predicted_Sequence' column.")
                except Exception as e:
                    print(f"     └─ ❌ Error reading {csv_path}: {e}")
        else:
            print(f"  📊 Prediction CSVs Found for {target_year}: 0 file(s) (Folder '{pred_year_dir}' does not exist)")

        # 3b. Add Vaccine
        if target_year in vaccine_dict:
            season_preds[f"Vaccine_{target_year}"] = vaccine_dict[target_year]
            print(f"  💉 Added Vaccine for {target_year}.")
        else:
            print(f"  ⚠️ No Vaccine FASTA mapped for {target_year}.")

        if not season_preds:
            print(f"  ⚠️ No predictions or vaccines available for {target_year}. Skipping evaluation.")
            continue

        season_boxplot_data = []

        # ── 4. Evaluate Globally for the Season (All Zones Unified) ───────────
        ref_seqs = season_ref_df['sequence'].tolist()
        test_seqs = season_test_df['sequence'].tolist()
        
        if not ref_seqs or not test_seqs: continue
        total_test_seqs = len(test_seqs)

        season_sequence_counts.append({
            'Season': target_year,
            'Reference_Start': ref_start.strftime('%Y-%m-%d'), 'Reference_End': ref_end.strftime('%Y-%m-%d'),
            'Reference_Count': len(ref_seqs),
            'Test_Start': test_start.strftime('%Y-%m-%d'), 'Test_End': test_end.strftime('%Y-%m-%d'),
            'Test_Count': total_test_seqs
        })
        
        for _, row in season_test_df.iterrows():
            evaluated_test_sequences.append({
                'Season': target_year, 'Original_Zone': row['zone'], 
                'ID': row['id'], 'Date': row['date'].strftime('%Y-%m-%d'), 'Sequence': row['sequence']
            })

        train_consensus = get_consensus(ref_seqs, seq_max_len)
        test_consensus = get_consensus(test_seqs, seq_max_len)
        test_pos_frequencies = get_positional_frequencies(test_seqs, seq_max_len)

        actual_shifts, actual_stables, season_pos_dict = set(), set(), {}
        for pos in range(seq_max_len):
            t_aa, g_aa = train_consensus[pos], test_consensus[pos]
            season_pos_dict[pos] = {'Season': target_year, 'Position': pos, 'Ref_Cons': t_aa, 'Test_Cons': g_aa}
            if t_aa != '-' and g_aa != '-':
                if t_aa != g_aa: actual_shifts.add(pos)
                else: actual_stables.add(pos)

        sorted_shifts = sorted(list(actual_shifts))
        test_population_motifs = ["".join([seq[p].upper() if p < len(seq) else '-' for p in sorted_shifts]) for seq in test_seqs]
        test_motif_counts = Counter(test_population_motifs)

        models_to_evaluate = season_preds.copy()
        models_to_evaluate[f"Representative_{target_year}"] = "".join(test_consensus)

        for model_name, pred_seq in models_to_evaluate.items():
            t_correct, t_valid, sh_correct, sh_valid, st_correct, st_valid, epi_correct, epi_valid = (0,)*8
            pred_motif = "".join([pred_seq[p].upper() if p < len(pred_seq) else '-' for p in sorted_shifts])
            
            for pos in range(seq_max_len):
                p_aa = pred_seq[pos].upper() if pos < len(pred_seq) else '-'
                season_pos_dict[pos][model_name] = p_aa
                
                valid_at_pos = sum(test_pos_frequencies[pos].values())
                correct_at_pos = test_pos_frequencies[pos].get(p_aa, 0)
                
                t_valid += valid_at_pos; t_correct += correct_at_pos
                
                if pos in actual_shifts:
                    sh_valid += valid_at_pos; sh_correct += correct_at_pos
                    all_shifted_positions_detailed.append({
                        'Season': target_year, 'Model': model_name, 'Position': pos,
                        'Ref_Consensus': train_consensus[pos], 'Test_Consensus': test_consensus[pos],
                        'Predicted_AA': p_aa, 'Match_Percent': f"{(correct_at_pos / valid_at_pos)*100:.2f}%" if valid_at_pos else "0%",
                        'Test_Distribution': get_aa_distribution(test_pos_frequencies, pos, total_test_seqs)
                    })
                if pos in actual_stables: st_valid += valid_at_pos; st_correct += correct_at_pos
                if pos in EPITOPE_0_BASED: epi_valid += valid_at_pos; epi_correct += correct_at_pos

            whole_seq_match_count = sum(1 for t_seq in test_seqs if pred_seq.upper()[:seq_max_len] == t_seq.upper()[:seq_max_len])
            
            # --- Sequence Level Distances ---
            motif_dists = get_sequence_distances(pred_seq, test_seqs, sorted_shifts)
            all_dists = get_sequence_distances(pred_seq, test_seqs, list(range(seq_max_len)))
            epi_dists = get_sequence_distances(pred_seq, test_seqs, list(EPITOPE_0_BASED))
            
            # --- Sequence Level Accuracies ---
            motif_accs = get_sequence_accuracies(pred_seq, test_seqs, sorted_shifts)
            all_accs = get_sequence_accuracies(pred_seq, test_seqs, list(range(seq_max_len)))
            epi_accs = get_sequence_accuracies(pred_seq, test_seqs, list(EPITOPE_0_BASED))
            
            m_min, m_median, m_mean, m_max = get_stats(motif_dists)
            a_min, a_median, a_mean, a_max = get_stats(all_dists)
            e_min, e_median, e_mean, e_max = get_stats(epi_dists)

            test_seq_ids = season_test_df['id'].tolist()
            for i, t_id in enumerate(test_seq_ids):
                # Save Raw BLOSUM Distances
                all_raw_distances.append({
                    'Season': target_year, 'Model': model_name, 'Test_Seq_ID': t_id, 
                    'AllPos_BLOSUM': all_dists[i], 'Motif_BLOSUM': motif_dists[i], 'Epitope_BLOSUM': epi_dists[i]
                })
                
                # Save Raw Accuracies
                all_raw_accuracies.append({
                    'Season': target_year, 'Model': model_name, 'Test_Seq_ID': t_id,
                    'AllPos_Accuracy': all_accs[i], 'Motif_Accuracy': motif_accs[i], 'Epitope_Accuracy': epi_accs[i]
                })
                
                season_boxplot_data.append({
                    'Model': model_name, 'AllPos_BLOSUM_Distance': all_dists[i]
                })

            # Append Summary
            all_summary_results.append({
                'Season': target_year, 'Model': model_name,
                'Total_Acc': f"{(t_correct/t_valid)*100:.2f}%" if t_valid else "0%",
                'Shift_Acc': f"{(sh_correct/sh_valid)*100:.2f}%" if sh_valid else "0%",
                'Epitope_Acc': f"{(epi_correct/epi_valid)*100:.2f}%" if epi_valid else "0%",
                'Whole_Seq_Match_Count': f"{whole_seq_match_count}/{total_test_seqs}",
                'Whole_Seq_Match_Pct': f"{(whole_seq_match_count/total_test_seqs)*100:.2f}%" if total_test_seqs else "0%",
                'Motif_BLOSUM_Mean': f"{m_mean:.2f}", 'AllPos_BLOSUM_Mean': f"{a_mean:.2f}", 'Epitope_BLOSUM_Mean': f"{e_mean:.2f}"
            })
            
            all_shift_motifs.append({
                'Season': target_year, 'Model': model_name, 'Num_Shifted_Sites': len(sorted_shifts), 'Predicted_Motif': pred_motif,
                'Exact_Motif_Match_Pop%': f"{(sum(count for pop_motif, count in test_motif_counts.items() if pop_motif == pred_motif)/total_test_seqs)*100:.2f}%",
                'Motif_BLOSUM_Mean': f"{m_mean:.2f}"
            })
            print(f"     - {model_name[:25]:<25} | All: {a_mean:.1f} | Motif: {m_mean:.1f} | Epi: {e_mean:.1f}")

        for pos in range(seq_max_len): all_position_predictions.append(season_pos_dict[pos])

        # Generate Season Boxplot (Global evaluation)
        if season_boxplot_data:
            df_plot = pd.DataFrame(season_boxplot_data)
            plt.figure(figsize=(16, 8))
            sns.boxplot(data=df_plot, x='Model', y='AllPos_BLOSUM_Distance', palette="Set2")
            plt.title(f"Season {target_year} - BLOSUM62 Distances by Model (All Zones Unified)", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{target_year}_BLOSUM_Boxplot.png"), dpi=300)
            plt.close()

    # ── 5. Save Output Files ──────────────────────────────────────────────────
    if all_summary_results:
        pd.DataFrame(all_summary_results).to_csv(os.path.join(out_dir, "1_Season_Summary.csv"), index=False)
        pd.DataFrame(all_shift_motifs).to_csv(os.path.join(out_dir, "2_Shift_Motifs.csv"), index=False)
        pd.DataFrame(all_shifted_positions_detailed).to_csv(os.path.join(out_dir, "3_Shifted_Positions_Detailed.csv"), index=False)
        pd.DataFrame(all_position_predictions).to_csv(os.path.join(out_dir, "4_Detailed_Position_Predictions.csv"), index=False)
        pd.DataFrame(all_raw_distances).to_csv(os.path.join(out_dir, "5_Raw_BLOSUM_Distances.csv"), index=False)
        pd.DataFrame(all_raw_accuracies).to_csv(os.path.join(out_dir, "6_Raw_Accuracies.csv"), index=False)  # <-- NEW
        pd.DataFrame(season_sequence_counts).to_csv(os.path.join(out_dir, "0_Season_Data_Counts.csv"), index=False)
        pd.DataFrame(evaluated_test_sequences).to_csv(os.path.join(out_dir, "0_Evaluated_Test_Sequences.csv"), index=False)
        print("\n" + "═"*70)
        print(f"✅ Evaluation complete. All datasets updated and saved to: {out_dir}")
        print("═"*70)
    else:
        print("\n⚠️ No evaluations were performed. Please check the logs above.")

if __name__ == "__main__":
    main()