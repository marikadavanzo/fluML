#!/usr/bin/env python3
"""
=================================================================
Usage:
    python evaluation_lv.py --base_dir h3n2_clean --fasta_file h3n2_clean.fasta --vaccine_dir Vaccine_strains

Description:
    Evaluates saved FASTA prediction files against the ground truth.
    Features: 
      - Population-weighted positional accuracy
      - Hamming & Levenshtein Distances for: Shifted, All, Epitope sites
      - Whole-Sequence Exact Match against viral swarm
      - Automatically evaluates representative test consensus sequence
      - Automatically loads and evaluates seasonal vaccine strains
      - Outputs detailed comparison between Hamming and Levenshtein distances
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
from datetime import datetime
import scipy.stats as stats
import Levenshtein



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

# ── Helpers ────────────────────────────────────────────────────────────

def load_fasta(fasta_path):
    print(f"Loading FASTA: {fasta_path}...")
    all_parsed_data = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        fields = record.description.split('|')
        parsed_date = None
        
        for field in fields:
            try:
                parsed_date = datetime.strptime(field.strip(), '%Y-%m-%d')
                break 
            except ValueError:
                continue
                
        if parsed_date is None:
            continue
            
        zone = fields[-1].strip()
        sequence = str(record.seq).replace('-', '').replace('*', '')
        
        all_parsed_data.append({
            'id': record.description, 
            'date': parsed_date,
            'zone': zone,
            'sequence': sequence
        })
        
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
    if pos not in pos_counts or total_seqs == 0: 
        return "None (0.00)"
    counts = pos_counts[pos]
    dist_strings = [f"{aa} ({count/total_seqs:.2f})" for aa, count in counts.most_common()]
    return ", ".join(dist_strings)

def parse_predicted_fasta(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if not line.startswith('>')]
        return "".join(lines).replace('-', '').replace('*', '').strip()
    except Exception:
        return None

def calc_levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1, s2)

def get_sequence_distances(pred_seq, test_seqs, positions_to_eval):
    hamming_distances = []
    levenshtein_distances = []
    
    # Pre-build the prediction substring once
    pred_sub = "".join([pred_seq[p].upper() if p < len(pred_seq) else '-' for p in positions_to_eval])
    
    for t_seq in test_seqs:
        # Build the target substring
        t_sub = "".join([t_seq[p].upper() if p < len(t_seq) else '-' for p in positions_to_eval])
        
        # Hamming Distance (Direct character comparison)
        h_dist = sum(1 for a, b in zip(pred_sub, t_sub) if a != b)
        hamming_distances.append(h_dist)
        
        # Levenshtein Distance
        l_dist = calc_levenshtein_distance(pred_sub, t_sub)
        levenshtein_distances.append(l_dist)
        
    return hamming_distances, levenshtein_distances

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
    if dist_list:
        return np.min(dist_list), np.median(dist_list), np.mean(dist_list), np.max(dist_list)
    return 0, 0, 0, 0

def calculate_proportional_pvalue(correct1, total1, correct2, total2):
    if total1 == 0 or total2 == 0: return 1.0
    wrong1, wrong2 = total1 - correct1, total2 - correct2
    try:
        _, p, _, _ = stats.chi2_contingency([[correct1, wrong1], [correct2, wrong2]])
        return p
    except ValueError:
        return 1.0

def calculate_continuous_pvalue(dist_array1, dist_array2):
    if len(dist_array1) == 0 or len(dist_array2) == 0: return 1.0
    if len(set(dist_array1) | set(dist_array2)) <= 1: return 1.0
    try:
        _, p = stats.mannwhitneyu(dist_array1, dist_array2, alternative='two-sided')
        return p
    except ValueError:
        return 1.0

def get_subset_metrics(pred_seq, subset_df, seq_max_len, sorted_shifts, actual_shifts):
    if len(subset_df) == 0: return None
    
    subset_seqs = subset_df['sequence'].tolist()
    pos_freqs = get_positional_frequencies(subset_seqs, seq_max_len)
    
    sh_valid, sh_correct = 0, 0
    for pos in range(seq_max_len):
        if pos in actual_shifts:
            p_aa = pred_seq[pos].upper() if pos < len(pred_seq) else '-'
            sh_valid += sum(pos_freqs[pos].values())
            sh_correct += pos_freqs[pos].get(p_aa, 0)
            
    all_dists_h, _ = get_sequence_distances(pred_seq, subset_seqs, list(range(seq_max_len)))
    _, _, a_mean, _ = get_stats(all_dists_h)
    
    return {
        'sh_correct': sh_correct, 
        'sh_valid': sh_valid,
        'Shift_Acc': (sh_correct/sh_valid)*100 if sh_valid else 0,
        'all_dists_raw': all_dists_h,
        'AllPos_Hamming_Mean': a_mean
    }

def evaluate_full_metrics(pred_seq, test_seqs, seq_max_len, sorted_shifts, actual_shifts, actual_stables, pos_freqs):
    t_correct, t_valid, sh_correct, sh_valid, st_correct, st_valid, epi_correct, epi_valid = (0,)*8
    
    if len(test_seqs) == 0: return None
        
    for pos in range(seq_max_len):
        p_aa = pred_seq[pos].upper() if pos < len(pred_seq) else '-'
        valid_at_pos = sum(pos_freqs[pos].values())
        correct_at_pos = pos_freqs[pos].get(p_aa, 0)
        
        t_valid += valid_at_pos; t_correct += correct_at_pos
        if pos in actual_shifts:
            sh_valid += valid_at_pos; sh_correct += correct_at_pos
        if pos in actual_stables: 
            st_valid += valid_at_pos; st_correct += correct_at_pos
        if pos in EPITOPE_0_BASED: 
            epi_valid += valid_at_pos; epi_correct += correct_at_pos

    whole_seq_match_count = sum(1 for t_seq in test_seqs if pred_seq.upper()[:seq_max_len] == t_seq.upper()[:seq_max_len])
    whole_seq_match_pct = (whole_seq_match_count / len(test_seqs)) * 100 if len(test_seqs) > 0 else 0
    
    all_positions = list(range(seq_max_len))
    epitope_positions = list(EPITOPE_0_BASED)
    
    motif_dists_h, motif_dists_l = get_sequence_distances(pred_seq, test_seqs, sorted_shifts)
    all_dists_h, all_dists_l = get_sequence_distances(pred_seq, test_seqs, all_positions)
    epi_dists_h, epi_dists_l = get_sequence_distances(pred_seq, test_seqs, epitope_positions)
    
    motif_accs = get_sequence_accuracies(pred_seq, test_seqs, sorted_shifts)
    all_accs = get_sequence_accuracies(pred_seq, test_seqs, all_positions)
    epi_accs = get_sequence_accuracies(pred_seq, test_seqs, epitope_positions)
    
    # Hamming Stats
    m_min, m_median, m_mean, m_max = get_stats(motif_dists_h)
    a_min, a_median, a_mean, a_max = get_stats(all_dists_h)
    e_min, e_median, e_mean, e_max = get_stats(epi_dists_h)
    
    # Levenshtein Stats
    m_min_l, m_median_l, m_mean_l, m_max_l = get_stats(motif_dists_l)
    a_min_l, a_median_l, a_mean_l, a_max_l = get_stats(all_dists_l)
    e_min_l, e_median_l, e_mean_l, e_max_l = get_stats(epi_dists_l)
    
    pred_motif = "".join([pred_seq[p].upper() if p < len(pred_seq) else '-' for p in sorted_shifts])
    test_motifs = ["".join([seq[p].upper() if p < len(seq) else '-' for p in sorted_shifts]) for seq in test_seqs]
    motif_match_exact = sum(1 for m in test_motifs if m == pred_motif)
    exact_motif_match_pct = (motif_match_exact / len(test_seqs)) * 100 if len(test_seqs) > 0 else 0
    
    return {
        'Total_Acc': f"{(t_correct/t_valid)*100:.2f}%" if t_valid else "0%",
        'Shift_Acc': f"{(sh_correct/sh_valid)*100:.2f}%" if sh_valid else "0%",
        'Epitope_Acc': f"{(epi_correct/epi_valid)*100:.2f}%" if epi_valid else "0%",
        'Whole_Seq_Match_Count': f"{whole_seq_match_count}/{len(test_seqs)}",
        'Whole_Seq_Match_Pct': f"{whole_seq_match_pct:.2f}%",
        
        # Hamming
        'Motif_Hamming_Mean': f"{m_mean:.2f}", 'Motif_Hamming_Median': m_median,
        'Motif_Hamming_Min': m_min, 'Motif_Hamming_Max': m_max,
        'AllPos_Hamming_Mean': f"{a_mean:.2f}", 'AllPos_Hamming_Median': a_median,
        'AllPos_Hamming_Min': a_min, 'AllPos_Hamming_Max': a_max,
        'Epitope_Hamming_Mean': f"{e_mean:.2f}", 'Epitope_Hamming_Median': e_median,
        'Epitope_Hamming_Min': e_min, 'Epitope_Hamming_Max': e_max,
        
        # Levenshtein
        'Motif_Levenshtein_Mean': f"{m_mean_l:.2f}", 'Motif_Levenshtein_Median': m_median_l,
        'Motif_Levenshtein_Min': m_min_l, 'Motif_Levenshtein_Max': m_max_l,
        'AllPos_Levenshtein_Mean': f"{a_mean_l:.2f}", 'AllPos_Levenshtein_Median': a_median_l,
        'AllPos_Levenshtein_Min': a_min_l, 'AllPos_Levenshtein_Max': a_max_l,
        'Epitope_Levenshtein_Mean': f"{e_mean_l:.2f}", 'Epitope_Levenshtein_Median': e_median_l,
        'Epitope_Levenshtein_Min': e_min_l, 'Epitope_Levenshtein_Max': e_max_l,
        
        'Exact_Motif_Match_Pop%': f"{exact_motif_match_pct:.2f}%",
        
        'raw_sh_correct': sh_correct, 'raw_sh_valid': sh_valid,
        'raw_all_dists': all_dists_h, 'raw_motif_dists': motif_dists_h, 'raw_epi_dists': epi_dists_h, 
        'raw_all_dists_l': all_dists_l, 'raw_motif_dists_l': motif_dists_l, 'raw_epi_dists_l': epi_dists_l, 
        'raw_all_accs': all_accs, 'raw_motif_accs': motif_accs, 'raw_epi_accs': epi_accs,
        'pred_motif': pred_motif
    }


MODELS_TO_SKIP = [
    "transformer_4l", 
    "transformer_5l", 
    "hotspot_reports", 
    "ensemble_entropyweighted", 
    "ensemble_majorityvote"
]

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Seasonal Shifts")
    parser.add_argument("--base_dir", type=str, required=True, help="Master backtest folder")
    parser.add_argument("--fasta_file", type=str, required=True, help="Master FASTA file")
    parser.add_argument("--vaccine_dir", type=str, default="Vaccine_strains", help="Directory containing seasonal vaccine strain FASTA files")
    args = parser.parse_args()

    fasta_basename = os.path.basename(args.fasta_file)
    fasta_name = os.path.splitext(fasta_basename)[0]
    out_dir = os.path.join(args.base_dir, f"{fasta_name}_LV")
    os.makedirs(out_dir, exist_ok=True)
    
    master_df = load_fasta(args.fasta_file)
    seq_max_len = int(master_df['sequence'].str.len().max())
    if 'zone' not in master_df.columns: master_df['zone'] = 'Global'
    
    season_folders = sorted([f for f in os.listdir(args.base_dir) if re.match(r"\d{4}_season", f)])

    all_summary_results = []
    zone_vs_overall = []
    all_shift_motifs = []
    all_shift_motifs_zones = []
    all_shifted_positions_detailed = []
    all_position_predictions = []
    all_raw_distances = [] 
    all_raw_accuracies = []
    season_sequence_counts = []
    evaluated_test_sequences = []
    
    dist_comparison_data = []
    diff_tracking_summary = {}

    for sf in season_folders:
        target_year = int(re.search(r"(\d{4})_season", sf).group(1))
        season_dir = os.path.join(args.base_dir, sf)
        diff_tracking_summary[target_year] = {'total_comparisons': 0, 'differences_found': 0}
        
        ref_start = pd.to_datetime(f"{target_year - 1}-07-01")
        ref_end   = pd.to_datetime(f"{target_year}-01-31")
        test_start= pd.to_datetime(f"{target_year}-07-01")
        test_end  = pd.to_datetime(f"{target_year + 1}-01-31")

        # FILTER TO UNIQUE PER-ZONE IMMEDIATELY
        ref_df = master_df[(master_df['date'] >= ref_start) & (master_df['date'] <= ref_end)].drop_duplicates(subset=['sequence', 'zone'])
        test_df = master_df[(master_df['date'] >= test_start) & (master_df['date'] <= test_end)].drop_duplicates(subset=['sequence', 'zone'])
        
        ref_seqs = ref_df['sequence'].tolist()
        test_seqs = test_df['sequence'].tolist()
        
        if not ref_seqs or not test_seqs: continue
            
        total_test_seqs = len(test_seqs)
        
        zones = [z for z in test_df['zone'].unique() if pd.notna(z)]
        zone_counts = {f"Zone_{z}_Unique": len(test_df[test_df['zone'] == z]) for z in zones}

        season_count_dict = {
            'Season': target_year,
            'Reference_Start': ref_start.strftime('%Y-%m-%d'),
            'Reference_End': ref_end.strftime('%Y-%m-%d'),
            'Reference_Count_Unique': len(ref_seqs),
            'Test_Start': test_start.strftime('%Y-%m-%d'),
            'Test_End': test_end.strftime('%Y-%m-%d'),
            'Test_Count_Unique_Per_Zone': total_test_seqs,
        }
        season_count_dict.update(zone_counts)
        season_sequence_counts.append(season_count_dict)
        
        for _, row in test_df.iterrows():
            evaluated_test_sequences.append({
                'Season': target_year, 'ID': row['id'],
                'Date': row['date'].strftime('%Y-%m-%d'), 'Sequence': row['sequence']
            })

        print(f"\nEvaluating Season: {target_year} ({total_test_seqs} unique per-zone seqs)")
        
        train_consensus = get_consensus(ref_seqs, seq_max_len)
        test_consensus = get_consensus(test_seqs, seq_max_len)
        test_pos_freqs = get_positional_frequencies(test_seqs, seq_max_len)

        actual_shifts = set()
        actual_stables = set()
        season_pos_dict = {}
        
        for pos in range(seq_max_len):
            t_aa, g_aa = train_consensus[pos], test_consensus[pos]
            season_pos_dict[pos] = {'Season': target_year, 'Position': pos, 'Ref_Cons': t_aa, 'Test_Cons': g_aa}
            if t_aa == '-' or g_aa == '-': continue
            if t_aa != g_aa: actual_shifts.add(pos)
            else: actual_stables.add(pos)

        sorted_shifts = sorted(list(actual_shifts))

        models_to_evaluate = {}
        models_to_evaluate[f"representative_target_{target_year}"] = "".join(test_consensus)
        models_to_evaluate[f"train_consensus_{target_year}"] = "".join(train_consensus)

        if os.path.exists(args.vaccine_dir):
            vaccine_patterns = [
                os.path.join(args.vaccine_dir, f"{target_year}*.fasta"),
                os.path.join(args.vaccine_dir, f"{target_year}*.fa"),
                os.path.join(args.vaccine_dir, f"{target_year}*.fna")
            ]
            for pat in vaccine_patterns:
                for v_file in sorted(glob.glob(pat)):
                    v_seq = parse_predicted_fasta(v_file)
                    if v_seq:
                        v_label = f"Vaccine_{os.path.splitext(os.path.basename(v_file))[0]}"
                        models_to_evaluate[v_label] = v_seq

        fasta_files = glob.glob(os.path.join(season_dir, "**", "*.fasta"), recursive=True)
        for fasta_path in fasta_files:
            if args.vaccine_dir in fasta_path: continue
                
            rel_path = os.path.relpath(fasta_path, season_dir)
            model_folder = rel_path.split(os.sep)[0]
            file_name = os.path.basename(fasta_path).replace(".fasta", "")
            model_name = file_name if "ensemble" in model_folder.lower() else model_folder

            if any(skip.lower() in model_name.lower() for skip in MODELS_TO_SKIP):
                continue

            parsed_seq = parse_predicted_fasta(fasta_path)
            if parsed_seq and model_name not in models_to_evaluate:
                models_to_evaluate[model_name] = parsed_seq

        for model_name, pred_seq in models_to_evaluate.items():
            res = evaluate_full_metrics(pred_seq, test_seqs, seq_max_len, sorted_shifts, actual_shifts, actual_stables, test_pos_freqs)
            if not res: continue

            for pos in range(seq_max_len):
                p_aa = pred_seq[pos].upper() if pos < len(pred_seq) else '-'
                season_pos_dict[pos][model_name] = p_aa
                
                if pos in actual_shifts:
                    valid_at_pos = sum(test_pos_freqs[pos].values())
                    correct_at_pos = test_pos_freqs[pos].get(p_aa, 0)
                    pop_match_pct = (correct_at_pos / valid_at_pos) * 100 if valid_at_pos > 0 else 0
                    all_shifted_positions_detailed.append({
                        'Season': target_year, 'Model': model_name, 'Position': pos,
                        'Ref_Consensus': train_consensus[pos], 'Test_Consensus': test_consensus[pos],
                        'Predicted_AA': p_aa, 'Match_Percent': f"{pop_match_pct:.2f}%",
                        'Test_Distribution': get_aa_distribution(test_pos_freqs, pos, total_test_seqs)
                    })

            summary_row = {'Season': target_year, 'Model': model_name}
            for k, v in res.items(): 
                if not k.startswith('raw_') and k != 'pred_motif': summary_row[k] = v
            all_summary_results.append(summary_row)
            
            for i, (_, row) in enumerate(test_df.iterrows()):
                h_all = res['raw_all_dists'][i]; l_all = res['raw_all_dists_l'][i]
                h_mot = res['raw_motif_dists'][i]; l_mot = res['raw_motif_dists_l'][i]
                h_epi = res['raw_epi_dists'][i]; l_epi = res['raw_epi_dists_l'][i]
                
                all_raw_distances.append({
                    'Season': target_year, 'Model': model_name, 'Test_Seq_ID': row['id'], 
                    'AllPos_Hamming': h_all, 'Shift_Hamming': h_mot, 'Epitope_Hamming': h_epi
                })
                
                dist_comparison_data.append({
                    'Season': target_year, 'Model': model_name, 'Test_Seq_ID': row['id'],
                    'AllPos_Hamming': h_all, 'AllPos_Levenshtein': l_all, 'AllPos_Diff': h_all - l_all,
                    'Motif_Hamming': h_mot, 'Motif_Levenshtein': l_mot, 'Motif_Diff': h_mot - l_mot,
                    'Epitope_Hamming': h_epi, 'Epitope_Levenshtein': l_epi, 'Epitope_Diff': h_epi - l_epi
                })
                
                if (h_all != l_all) or (h_mot != l_mot) or (h_epi != l_epi):
                    diff_tracking_summary[target_year]['differences_found'] += 1
                diff_tracking_summary[target_year]['total_comparisons'] += 1
                
                all_raw_accuracies.append({
                    'Season': target_year, 'Model': model_name, 'Test_Seq_ID': row['id'],
                    'AllPos_Accuracy': res['raw_all_accs'][i],
                    'Motif_Accuracy': res['raw_motif_accs'][i],
                    'Epitope_Accuracy': res['raw_epi_accs'][i]
                })

            all_shift_motifs.append({
                'Season': target_year, 'Model': model_name, 'Num_Shifted_Sites': len(sorted_shifts),
                'Predicted_Motif': res['pred_motif'],
                'Exact_Motif_Match_Pop%': res['Exact_Motif_Match_Pop%'],
                'Motif_Hamming_Mean': res['Motif_Hamming_Mean'], 'Motif_Hamming_Median': res['Motif_Hamming_Median'],
                'Motif_Hamming_Min': res['Motif_Hamming_Min'], 'Motif_Hamming_Max': res['Motif_Hamming_Max'],
            })
            print(f"    - {model_name[:25]:<25} | Mean Dist (Hamming): {res['AllPos_Hamming_Mean']}")

            for z in zones:
                z_df = test_df[test_df['zone'] == z]
                z_freq = get_positional_frequencies(z_df['sequence'].tolist(), seq_max_len)
                z_res = evaluate_full_metrics(pred_seq, z_df['sequence'].tolist(), seq_max_len, sorted_shifts, actual_shifts, actual_stables, z_freq)
                
                if z_res:
                    zone_vs_overall.append({
                        'Season': target_year, 'Model': model_name, 'Zone': z,
                        'Zone_Shift_Acc': z_res['Shift_Acc'], 'Overall_Shift_Acc': res['Shift_Acc'],
                        'Zone_Dist_Mean': z_res['AllPos_Hamming_Mean'], 'Overall_Dist_Mean': res['AllPos_Hamming_Mean'],
                    })
                    all_shift_motifs_zones.append({
                        'Season': target_year, 'Model': model_name, 'Zone': z,
                        'Predicted_Motif': res['pred_motif'],
                        'Zone_Exact_Match%': z_res['Exact_Motif_Match_Pop%'],
                        'Zone_Motif_Mean': z_res['Motif_Hamming_Mean']
                    })

        for pos in range(seq_max_len):
            all_position_predictions.append(season_pos_dict[pos])

    # --- Save Output DataFrames ---
    pd.DataFrame(all_summary_results).to_csv(os.path.join(out_dir, "1a_Season_Summary.csv"), index=False)
    pd.DataFrame(zone_vs_overall).to_csv(os.path.join(out_dir, "1b_Zone_vs_Overall.csv"), index=False)
    pd.DataFrame(all_shift_motifs).to_csv(os.path.join(out_dir, "2a_Shift_Motifs_Overall.csv"), index=False)
    pd.DataFrame(all_shift_motifs_zones).to_csv(os.path.join(out_dir, "2b_Shift_Motifs_Zones.csv"), index=False)
    pd.DataFrame(all_shifted_positions_detailed).to_csv(os.path.join(out_dir, "3_Shifted_Positions_Detailed.csv"), index=False)
    pd.DataFrame(all_position_predictions).to_csv(os.path.join(out_dir, "4_Detailed_Position_Predictions.csv"), index=False)
    pd.DataFrame(all_raw_distances).to_csv(os.path.join(out_dir, "5_Raw_Hamming_Distances.csv"), index=False)
    pd.DataFrame(all_raw_accuracies).to_csv(os.path.join(out_dir, "6_Raw_Accuracies.csv"), index=False)
    pd.DataFrame(season_sequence_counts).to_csv(os.path.join(out_dir, "0_Season_Data_Counts.csv"), index=False)
    pd.DataFrame(evaluated_test_sequences).to_csv(os.path.join(out_dir, "0_Evaluated_Test_Sequences.csv"), index=False)
    pd.DataFrame(dist_comparison_data).to_csv(os.path.join(out_dir, "7_Hamming_vs_Levenshtein_Comparison.csv"), index=False)
    
    txt_path = os.path.join(out_dir, "7_Hamming_vs_Levenshtein_Summary.txt")
    with open(txt_path, "w") as f:
        f.write("Hamming vs. Levenshtein Differences Summary\n")
        f.write("===========================================\n")
        total_diffs = 0; total_comps = 0
        for season, counts in diff_tracking_summary.items():
            diffs = counts['differences_found']; comps = counts['total_comparisons']
            f.write(f"Season {season}: {diffs} differences found across {comps} test sequence comparisons.\n")
            total_diffs += diffs; total_comps += comps
            
        f.write("\n-------------------------------------------\n")
        f.write(f"Total across all seasons: {total_diffs} differences out of {total_comps} sequence comparisons.\n")

    print(f"\n✅ Evaluation complete. All datasets updated and saved to: {out_dir}")

if __name__ == "__main__":
    main()