#!/usr/bin/env python3
"""

Extracts what a saved model "understands". Upgraded to support:
1. Automated discovery of "best" checkpoints across all seasons.
2. Batch processing of actual training sequences (FASTA) with dynamic season filtering.
3. Future clade sequence trajectory forecasting.
(Note: Native attention epistasis removed as the model architecture does not support it).

python inspection.py --base_dir h3n2_clean --train_fasta h3n2_clean.fasta 
"""

import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pickle
import argparse
from collections import Counter
from tqdm import tqdm

import warnings
import math
from typing import Dict, List, Optional, Tuple
import csv
import re
import gc
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import EsmTokenizer
from Bio import SeqIO
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pickle


# Import model builders and custom layers in case load_model requires them
from models_ym_clade import build_transformer_model, build_LSTM_model

from models_ym_clade import (
    load_fasta,
    prepare_data,
    tokenize_sequences,
    build_epitope_mask,
    create_padding_mask_logic,
    create_look_ahead_mask_logic,
    compute_fixed_positions,
    compute_position_entropy,
    build_position_loss_weights,
    make_padding_mask,
    make_look_ahead_mask,
    make_combined_mask,
    get_reference_tokens,
    _build_window_schedule,
    PositionContextGating,
    TimeDecayLayer,
    SpatioTemporalFiLM
)

MODEL_NAME = 'facebook/esm2_t12_35M_UR50D'
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

try:
    from Bio.Align import substitution_matrices
    BLOSUM62 = substitution_matrices.load("BLOSUM62")
    STANDARD_AAS = [a for a in BLOSUM62.alphabet if a.isalpha() and a not in ('B', 'Z', 'X', 'J', 'U', 'O')]
except Exception:
    BLOSUM62 = None
    STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")


# ============================================================================
# Discovery and Loading Helpers
# ============================================================================

def find_best_models(base_dir):
    """Scans directories to find the best checkpoints, ignoring '_last.keras'."""
    best_models = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".keras") and not file.endswith("_last.keras"):
                model_path = os.path.join(root, file)
                assets_path = os.path.join(root, "model_assets.pkl")
                if os.path.exists(assets_path):
                    parts = root.replace("\\", "/").split("/")
                    variant_name = parts[-1]
                    season = parts[-2] if len(parts) > 1 else "unknown_season"
                    best_models.append((season, variant_name, model_path, assets_path))
    return best_models

def load_assets(assets_pkl):
    with open(assets_pkl, "rb") as f:
        return pickle.load(f)

def _route_inputs(model, t_dict, dec_input):
    """Dynamically routes the correct number of inputs depending on Transformer vs LSTM."""
    if len(model.inputs) == 10:
        # Transformer
        return [
            t_dict["enc_in"], t_dict["time_in"], t_dict["zone_in"], t_dict["clade_in"], 
            t_dict["prop_in"], t_dict["epi_in"], dec_input, 
            t_dict["pos_in_enc"], t_dict["pos_in_dec"], t_dict["ref_in"]
        ]
    else:
        # LSTM (Omits pos_in_enc and pos_in_dec)
        return [
            t_dict["enc_in"], t_dict["time_in"], t_dict["zone_in"], t_dict["clade_in"], 
            t_dict["prop_in"], t_dict["epi_in"], dec_input, 
            t_dict["ref_in"]
        ]
def load_fasta(fasta_path):
    """Loads a master FASTA file and extracts dates from the headers."""
    print(f"Loading master FASTA: {fasta_path}...")
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
        if parsed_date is None:
            continue
        sequence = str(record.seq).replace('-', '').replace('*', '')
        all_parsed_data.append({'id': record.id, 'date': parsed_date, 'sequence': sequence})
        
    df = pd.DataFrame(all_parsed_data)
    print(f"✅ Loaded {len(df)} valid sequences.")
    return df

def find_layer(model, name_substring):
    matches = [l for l in model.layers if name_substring.lower() in l.name.lower()]
    if not matches:
        raise ValueError(f"No layer matching '{name_substring}' found.")
    return matches[0]

def get_consensus(seq_list, seq_max_len):
    """Generates the most frequent amino acid per position."""
    consensus = []
    for pos in range(seq_max_len):
        aas = [s[pos] for s in seq_list if pos < len(s) and s[pos] not in ['-', 'X', '<PAD>']]
        consensus.append(str(Counter(aas).most_common(1)[0][0]).strip().upper() if aas else '-')
    return consensus


# ============================================================================
# A. Zero-input weight introspection
# ============================================================================

def a1_learned_substitution_geometry(model, tokenizer, out_dir):
    embed_layer = None
    for l in model.layers:
        if isinstance(l, tf.keras.layers.Embedding) and l.input_dim == tokenizer.vocab_size:
            embed_layer = l
            break
    if embed_layer is None:
        return

    W = embed_layer.get_weights()[0]
    aa_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AAS}
    aa_ids = {aa: i for aa, i in aa_ids.items() if i is not None}
    aas = list(aa_ids.keys())
    vecs = np.stack([W[aa_ids[aa]] for aa in aas])

    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    learned_sim = norm @ norm.T
    pd.DataFrame(learned_sim, index=aas, columns=aas).to_csv(os.path.join(out_dir, "a1_learned_aa_similarity.csv"))

    plt.figure(figsize=(10, 8))
    plt.imshow(learned_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.xticks(range(len(aas)), aas); plt.yticks(range(len(aas)), aas)
    plt.colorbar(label="Cosine similarity")
    plt.title("Learned AA substitution geometry")
    plt.savefig(os.path.join(out_dir, "a1_learned_aa_similarity.png"), dpi=150)
    plt.close()

def a2_zone_embedding_clustering(model, zone_to_id, out_dir):
    try:
        layer = find_layer(model, "zone_embed")
    except ValueError: return
    W = layer.get_weights()[0]
    id_to_zone = {v: k for k, v in zone_to_id.items()}
    zones = [id_to_zone[i] for i in range(W.shape[0]) if i in id_to_zone]
    vecs = W[[zone_to_id[z] for z in zones]]

    dist = squareform(pdist(vecs, metric="cosine"))
    pd.DataFrame(dist, index=zones, columns=zones).to_csv(os.path.join(out_dir, "a2_zone_embedding_distance.csv"))

    Z = linkage(pdist(vecs, metric="cosine"), method="average")
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=zones, leaf_rotation=90)
    plt.title("Zone embedding clustering")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "a2_zone_dendrogram.png"), dpi=150)
    plt.close()

def a3_clade_embedding_clustering(model, assets, out_dir):
    """Clusters the learned clade embeddings to see how the model groups viral lineages."""
    clade_to_id = assets.get("clade_to_id", {})
    if not clade_to_id:
        print("      -> [SKIPPED a3] 'clade_to_id' not found in assets or is empty.")
        return 

    try:
        layer = find_layer(model, "clade_embed")
    except ValueError: 
        print("      -> [SKIPPED a3] Layer named 'clade_embed' not found in the model.")
        return
        
    W = layer.get_weights()[0]
    id_to_clade = {v: k for k, v in clade_to_id.items()}
    clades = [id_to_clade[i] for i in range(W.shape[0]) if i in id_to_clade]
    vecs = W[[clade_to_id[c] for c in clades]]

    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram

    dist = squareform(pdist(vecs, metric="cosine"))
    pd.DataFrame(dist, index=clades, columns=clades).to_csv(os.path.join(out_dir, "a3_clade_embedding_distance.csv"))

    # Only plot if we have more than 1 clade
    if len(clades) > 1:
        Z = linkage(pdist(vecs, metric="cosine"), method="average")
        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=clades, leaf_rotation=90)
        plt.title("Learned Clade Phylogenetic Grouping (Embeddings)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "a3_clade_dendrogram.png"), dpi=150)
        plt.close()
        print(f"      -> Saved a3 clade clustering for {len(clades)} clades.")
    else:
        print("      -> [WARNING a3] CSV saved, but skipped dendrogram PNG because there is only 1 clade.")

def a4_contextual_substitution_geometry(model, tokenizer, assets, train_seqs, zone_id, clade_id, out_dir):
    print("      -> Extracting contextual representations from deeper layers...")
    standard_aas = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
    
    # Locate final hidden state layer
    target_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 3 and layer.output_shape[-1] != tokenizer.vocab_size:
            target_layer = layer
            break
            
    if not target_layer:
        print("      -> [ERROR] Could not find the final transformer block.")
        return
        
    extractor = tf.keras.models.Model(inputs=model.inputs, outputs=target_layer.output)
    
    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    max_len_target = assets["MAX_LEN_TARGET"]
    
    # Sample up to 50 training sequences for context extraction
    sample_seqs = train_seqs[:50] 
    tokens = [tokenizer.encode(seq, add_special_tokens=True) for seq in sample_seqs]
    batch_tokens = pad_sequences(tokens, maxlen=max_len_input, padding="post", value=tokenizer.pad_token_id)
    dec_tokens = pad_sequences([t[:max_len_target] for t in tokens], maxlen=max_len_target, padding="post", value=tokenizer.pad_token_id)
    dec_input = tf.constant(dec_tokens, dtype="int32")

    t = _build_batch_tensors(assets, tokenizer, batch_tokens, zone_id, clade_id, delta_months=0.0)
    inputs = _route_inputs(model, t, dec_input)
    
    hidden_states = extractor.predict(inputs, batch_size=16, verbose=0)
    
    aa_vectors = {aa: [] for aa in standard_aas}
    max_pos = min(batch_tokens.shape[1], hidden_states.shape[1])
    
    for b_idx in range(len(sample_seqs)):
        seq_tokens = batch_tokens[b_idx]
        for pos_idx in range(max_pos):
            token_id = seq_tokens[pos_idx]
            aa = tokenizer.convert_ids_to_tokens(int(token_id))
            if aa in standard_aas:
                aa_vectors[aa].append(hidden_states[b_idx, pos_idx, :])
                
    aa_means = {
        aa: (np.mean(aa_vectors[aa], axis=0) if len(aa_vectors[aa]) > 0 else np.zeros(hidden_states.shape[-1]))
        for aa in standard_aas
    }
            
    vecs = np.stack([aa_means[aa] for aa in standard_aas])
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    learned_sim = norm @ norm.T
    
    # 1. Save the CSV for your seasonal stability script
    df_learned = pd.DataFrame(learned_sim, index=standard_aas, columns=standard_aas)
    df_learned.to_csv(os.path.join(out_dir, "a4_contextual_aa_similarity.csv"))
    
    # 2. (Optional) Save a 20x20 heatmap just like a1 does
    plt.figure(figsize=(10, 8))
    plt.imshow(learned_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.xticks(range(len(standard_aas)), standard_aas)
    plt.yticks(range(len(standard_aas)), standard_aas)
    plt.colorbar(label="Cosine similarity")
    plt.title("Contextual AA substitution geometry (Last Layer)")
    plt.savefig(os.path.join(out_dir, "a4_contextual_aa_similarity.png"), dpi=150)
    plt.close()

def b1_epistasis_scan_masked_probe(model, tokenizer, assets, base_sequence, train_sequences, zone_to_id, clade_id, delta_months, positions_to_test, out_dir):
    """
    Masked Knockout Probe (Option B) - POOLED GLOBAL VERSION:
    Averages the epistatic reaction across all zones to create a global structural map.
    """
    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    max_len_target = assets["MAX_LEN_TARGET"]
    seq_len = len(base_sequence)

    print("      -> Building historical amino acid dictionary...")
    historical_aas = {pos: set() for pos in range(seq_len)}
    for seq in train_sequences:
        for pos, aa in enumerate(seq):
            if pos < seq_len and aa not in ['-', 'X']:
                historical_aas[pos].add(aa.upper())

    x_id = tokenizer.convert_tokens_to_ids('X')
    if x_id is None: 
        x_id = tokenizer.pad_token_id

    base_tokens = tokenizer.encode("".join(base_sequence), add_special_tokens=True)
    base_padded = pad_sequences([base_tokens], maxlen=max_len_input, padding="post", value=tokenizer.pad_token_id)
    base_dec_tokens = pad_sequences([base_tokens[:max_len_target]], maxlen=max_len_target, padding="post", value=tokenizer.pad_token_id)
    
    # =====================================================================
    # THE SPEED FIX: Compile the forward pass to bypass Python overhead
    # =====================================================================
    @tf.function(reduce_retracing=True)
    def fast_predict_step(enc_in, time_in, zone_in, clade_in, prop_in, epi_in, dec_input, pos_in_enc, pos_in_dec, ref_in):
        if len(model.inputs) == 10:
            inputs = [enc_in, time_in, zone_in, clade_in, prop_in, epi_in, dec_input, pos_in_enc, pos_in_dec, ref_in]
        else:
            # LSTM drops the positional encodings
            inputs = [enc_in, time_in, zone_in, clade_in, prop_in, epi_in, dec_input, ref_in]
        logits = model(inputs, training=False)
        return tf.nn.softmax(logits[0])

    def get_probs_global(enc_tokens_batch, dec_tokens_batch):
        all_zone_probs = []
        dec_input = tf.constant(dec_tokens_batch, dtype="int32")
        
        for zone_name, z_id in zone_to_id.items():
            t = _build_batch_tensors(assets, tokenizer, enc_tokens_batch, z_id, clade_id, delta_months)
            
            # Call the compiled TF function instead of the raw model
            probs = fast_predict_step(
                t["enc_in"], t["time_in"], t["zone_in"], t["clade_in"], 
                t["prop_in"], t["epi_in"], dec_input, 
                t["pos_in_enc"], t["pos_in_dec"], t["ref_in"]
            )
            all_zone_probs.append(probs.numpy())
            
        return np.mean(all_zone_probs, axis=0)
    # =====================================================================

    records = []
    STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

    print(f"      -> Running Global Masked Probe on {len(positions_to_test)} positions across {len(zone_to_id)} zones...")
    for pos_i in tqdm(positions_to_test, desc="Scanning Positions"):
        if pos_i >= seq_len: continue
        orig_aa = base_sequence[pos_i].upper()
        
        masked_base_dec = base_dec_tokens.copy()
        if (pos_i + 2) < max_len_target:
            masked_base_dec[0, pos_i + 2:] = x_id
            
        # Get baseline probabilities (Averaged globally)
        baseline_probs_masked = get_probs_global(base_padded, masked_base_dec)
        
        for mut_aa in STANDARD_AAS:
            if mut_aa == orig_aa: continue
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            if mut_id is None: continue
                
            mutated_enc = base_padded.copy()
            mutated_enc[0, pos_i + 1] = mut_id 
            
            mutated_dec = masked_base_dec.copy()
            if (pos_i + 1) < max_len_target:
                mutated_dec[0, pos_i + 1] = mut_id
            
            # Get new mutated probabilities (Averaged globally)
            mut_probs_masked = get_probs_global(mutated_enc, mutated_dec)
            
            delta = np.abs(mut_probs_masked - baseline_probs_masked).sum(axis=-1) 
            is_historical = "Yes" if mut_aa in historical_aas[pos_i] else "No"
            
            for pos_j in range(seq_len):
                if pos_j == pos_i: continue
                
                if delta[pos_j] > 0.01:  
                    records.append({
                        "Mutated_Pos": pos_i + 1, 
                        "Original_AA": orig_aa,
                        "Mutation": mut_aa, 
                        "Historically_Observed": is_historical,
                        "Affected_Pos": pos_j + 1,
                        "Distance_1D": abs(pos_i - pos_j),
                        "Impact_Score": float(delta[pos_j])
                    })

    # ... (CSV and Plot saving code remains identical)
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Impact_Score", ascending=False)
        df.to_csv(os.path.join(out_dir, "b1_epistasis_pairs_masked.csv"), index=False)
        
        pivot = df.pivot_table(index="Mutated_Pos", columns="Affected_Pos", values="Impact_Score", aggfunc="max").fillna(0)
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot.values, cmap="magma", aspect="auto")
        plt.colorbar(label="Predicted Probability Shift (L1)")
        plt.xlabel("Affected Position")
        plt.ylabel("Mutated Position")
        plt.title("Long-Range Epistatic Impact Map (Global Average)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "b1_epistasis_heatmap_masked.png"), dpi=150)
        plt.close()
        
        far_interactions = len(df[df["Distance_1D"] > 5])
        print(f"      -> Found {len(df)} significant pairs ({far_interactions} are long-range / 3D).")
    else:
        print("      -> No significant epistatic shifts detected.")

        
def b1_epistasis_scan(model, tokenizer, assets, base_sequence, zone_id, clade_id, delta_months, positions_to_test, out_dir):
    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    max_len_target = assets["MAX_LEN_TARGET"]
    seq_len = len(base_sequence)

    base_tokens = tokenizer.encode("".join(base_sequence), add_special_tokens=True)
    base_padded = pad_sequences([base_tokens], maxlen=max_len_input, padding="post", value=tokenizer.pad_token_id)
    
    # Base decoder input (typically shifted right by 1, starting with a BOS token)
    base_dec_tokens = pad_sequences([base_tokens[:max_len_target]], maxlen=max_len_target, padding="post", value=tokenizer.pad_token_id)
    
    def get_probs(enc_tokens_batch, dec_tokens_batch):
        t = _build_batch_tensors(assets, tokenizer, enc_tokens_batch, zone_id, clade_id, delta_months)
        
        # Cast the dynamically updated decoder tokens into a tensor
        dec_input = tf.constant(dec_tokens_batch, dtype="int32")
        
        inputs = _route_inputs(model, t, dec_input)
        
        logits = model(inputs, training=False).numpy()
        return tf.nn.softmax(logits[0]).numpy()

    # Get baseline probabilities using the unmutated sequence
    baseline_probs = get_probs(base_padded, base_dec_tokens)
    records = []

    print(f"      -> Scanning {len(positions_to_test)} positions for epistatic reactions...")
    for pos_i in positions_to_test:
        if pos_i >= seq_len: continue
        orig_aa = base_sequence[pos_i]
        
        
        for mut_aa in STANDARD_AAS:
            if mut_aa == orig_aa: continue
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            if mut_id is None: continue
                
            # 1. Mutate the Encoder Input
            mutated_enc = base_padded.copy()
            mutated_enc[0, pos_i + 1] = mut_id  # +1 accounts for the <CLS> or <BOS> token
            
            # 2. Mutate the Decoder Input (CRITICAL FIX)
            mutated_dec = base_dec_tokens.copy()
            if (pos_i + 1) < max_len_target:
                mutated_dec[0, pos_i + 1] = mut_id
            
            # 3. Get new probabilities with the mutation visible to the whole network
            mut_probs = get_probs(mutated_enc, mutated_dec)
            
            # Calculate probability shift (L1 distance across the vocabulary per position)
            delta = np.abs(mut_probs - baseline_probs).sum(axis=-1) 
            
            for pos_j in range(seq_len):
                if pos_j == pos_i: continue
                
                if delta[pos_j] > 0.01:  
                    records.append({
                        "Mutated_Pos": pos_i + 1, 
                        "Mutation": mut_aa, 
                        "Affected_Pos": pos_j + 1,
                        "Impact_Score": float(delta[pos_j])
                    })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Impact_Score", ascending=False)
        df.to_csv(os.path.join(out_dir, "b1_epistasis_pairs.csv"), index=False)
        
        pivot = df.pivot_table(index="Mutated_Pos", columns="Affected_Pos", values="Impact_Score", aggfunc="max").fillna(0)
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot.values, cmap="magma", aspect="auto")
        plt.colorbar(label="Predicted Probability Shift (L1)")
        plt.xlabel("Affected Position")
        plt.ylabel("Mutated Position")
        plt.title("Learned Epistatic Impact Map")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "b1_epistasis_heatmap.png"), dpi=150)
        plt.close()
        print(f"      -> Found {len(df)} significant epistatic pairs.")
    else:
        print("      -> No significant epistatic shifts detected above 1% threshold.")

def t1_temporal_sensitivity(model, tokenizer, assets, sequences, zone_id, clade_id, out_dir):
    print(f"      -> Running batch-optimized temporal sensitivity on {len(sequences)} sequences...")
    
    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    max_len_target = assets["MAX_LEN_TARGET"]
    
    tokens = [tokenizer.encode(seq, add_special_tokens=True) for seq in sequences]
    batch_tokens = pad_sequences(tokens, maxlen=max_len_input, padding="post", value=tokenizer.pad_token_id)
    
    dec_tokens = pad_sequences([t[:max_len_target] for t in tokens], maxlen=max_len_target, padding="post", value=tokenizer.pad_token_id)
    dec_input = tf.constant(dec_tokens, dtype="int32")

    # Get Baseline (Month 0) - USING .predict() with batch_size
    t0 = _build_batch_tensors(assets, tokenizer, batch_tokens, zone_id, clade_id, 0.0)
    inputs0 = _route_inputs(model, t0, dec_input)
    
    
    baseline_logits = model.predict(inputs0, batch_size=32, verbose=0)
    baseline_probs = tf.nn.softmax(baseline_logits).numpy()

    months_to_test = list(range(0, 25))
    results = []

    for m in tqdm(months_to_test, desc="Analyzing temporal sensitivity"):
        t = _build_batch_tensors(assets, tokenizer, batch_tokens, zone_id, clade_id, float(m))
        inputs = _route_inputs(model, t, dec_input)
        
        # USING .predict() with batch_size
        future_logits = model.predict(inputs, batch_size=32, verbose=0)
        future_probs = tf.nn.softmax(future_logits).numpy()
        
        variation = np.mean(np.abs(baseline_probs - future_probs))
        results.append({"Delta_Months": m, "Mean_Shift": variation})

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(out_dir, "t1_temporal_drift_batch.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df_results["Delta_Months"], df_results["Mean_Shift"], marker='o', linestyle='-', color='r')
    plt.xlabel("Delta Months into the Future")
    plt.ylabel("Mean Probability Shift (Batch Average)")
    plt.title("Population Temporal Sensitivity (Batch Optimized)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "t1_temporal_drift_batch.png"), dpi=150)
    plt.close()


def t2_historical_decay_probe(model, tokenizer, assets, sequence, zone_id, clade_id, out_dir):
    """
    Probes the time decay by extracting the specific TimeDecayLayer output, 
    bypassing LayerNorms that wash out the magnitude.
    """
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import os
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Model

    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    max_len_target = assets["MAX_LEN_TARGET"]
    max_delta = assets["MAX_DELTA_MONTHS"]
    
    base_tokens = tokenizer.encode("".join(sequence), add_special_tokens=True)
    batch_tokens = pad_sequences([base_tokens], maxlen=max_len_input, padding="post", value=tokenizer.pad_token_id)
    
    # 1. Find the TimeDecayLayer in the model
    decay_layer_name = None
    for l in model.layers:
        if "time_decay" in l.name.lower():
            decay_layer_name = l.name
            break
            
    if not decay_layer_name:
        print("      -> [WARNING] TimeDecayLayer not found. Skipping T2.")
        return

    # 2. Create a sub-model that outputs ONLY the TimeDecayLayer tensor
    extractor = Model(inputs=model.inputs, outputs=model.get_layer(decay_layer_name).output)

    ages_to_test = list(range(0, 37)) 
    signal_strengths = []
    
    for age in ages_to_test:
        simulated_d_months = max(0.0, max_delta - age)
        
        t = _build_batch_tensors(assets, tokenizer, batch_tokens, zone_id, clade_id, simulated_d_months)
        
        # We need a dummy decoder input to satisfy the model signature, even though we extract early
        dec_input = tf.zeros((1, max_len_target), dtype="int32")
        
        inputs = _route_inputs(model, t, dec_input)
        
        # Get the decayed embedding
        decayed_vector = extractor(inputs, training=False).numpy()
        
        # Calculate L2 Norm of just the time embedding
        signal_magnitude = np.linalg.norm(decayed_vector)
        signal_strengths.append(signal_magnitude)

    # Normalize to Month 0
    max_strength = max(signal_strengths)
    if max_strength > 0:
        relative_strengths = [s / max_strength for s in signal_strengths]
    else:
        relative_strengths = signal_strengths

    plt.figure(figsize=(8, 5))
    plt.plot(ages_to_test, relative_strengths, marker='o', linestyle='-', color='teal')
    plt.fill_between(ages_to_test, relative_strengths, alpha=0.2, color='teal')
    
    plt.xlabel("Historical Age of Sequence (Months)")
    plt.ylabel("Relative Time Embedding Strength")
    plt.title("Isolated TimeDecayLayer Output Magnitude")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "t2_isolated_historical_decay.png"), dpi=150)
    plt.close()
    
    print(f"      -> Analyzed isolated time decay layer. Saved to {out_dir}/t2_isolated_historical_decay.png")
    
# ============================================================================
# C. Context-effect decomposition (Batch sequences)
# ============================================================================

def _build_batch_tensors(assets, tokenizer, batch_tokens, zone_id, clade_id, delta_months):
    batch_size = batch_tokens.shape[0]
    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    max_len_target = assets["MAX_LEN_TARGET"]

    enc_in = tf.constant(batch_tokens, dtype="int32")
    ref_in = tf.constant(batch_tokens, dtype="int32")
    epi_in = tf.zeros((batch_size, max_len_input), dtype="int32")
    prop_in = tf.zeros((batch_size, max_len_input, 2), dtype="float32")

    time_vec = np.array([[float(delta_months), 0.0, 1.0]], dtype="float32")
    time_in = tf.constant(np.tile(time_vec, (batch_size, 1)))

    zone_in = tf.constant(np.tile([[zone_id]], (batch_size, 1)), dtype="int32")
    clade_in = tf.constant(np.tile([[clade_id]], (batch_size, 1)), dtype="int32")

    pos_in_enc = tf.constant(np.tile(np.arange(max_len_input)[np.newaxis, :], (batch_size, 1)), dtype="int32")
    pos_in_dec = tf.constant(np.tile(np.arange(max_len_target)[np.newaxis, :], (batch_size, 1)), dtype="int32")

    return dict(enc_in=enc_in, ref_in=ref_in, epi_in=epi_in, prop_in=prop_in,
                time_in=time_in, zone_in=zone_in, clade_in=clade_in,
                pos_in_enc=pos_in_enc, pos_in_dec=pos_in_dec)

def c1_zone_divergence_map(model, tokenizer, assets, sequences, zone_to_id, clade_id, delta_months, out_dir):
    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    tokens = [tokenizer.encode(seq, add_special_tokens=True) for seq in sequences]
    batch_tokens = pad_sequences(tokens, maxlen=max_len_input, padding="post", value=1)
    dec_input = tf.zeros((len(sequences), assets["MAX_LEN_TARGET"]), dtype="int32")

    zone_probs = {}
    for zone, zid in zone_to_id.items():
        t = _build_batch_tensors(assets, tokenizer, batch_tokens, zid, clade_id, delta_months)
        inputs = _route_inputs(model, t, dec_input)
        logits = model.predict(inputs, batch_size=32, verbose=0)
        zone_probs[zone] = np.mean(tf.nn.softmax(logits).numpy(), axis=0)

    zones = list(zone_probs.keys())
    n = len(zones)
    div = np.zeros((n, n))
    for i, zi in enumerate(zones):
        for j, zj in enumerate(zones):
            p, q = zone_probs[zi], zone_probs[zj]
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log((p + 1e-9) / (m + 1e-9)), axis=-1)
            kl_qm = np.sum(q * np.log((q + 1e-9) / (m + 1e-9)), axis=-1)
            div[i, j] = np.mean(0.5 * kl_pm + 0.5 * kl_qm)

    pd.DataFrame(div, index=zones, columns=zones).to_csv(os.path.join(out_dir, "c1_zone_divergence.csv"))
    
    plt.figure(figsize=(9, 7))
    plt.imshow(div, cmap="viridis")
    plt.xticks(range(n), zones, rotation=45); plt.yticks(range(n), zones)
    plt.colorbar(label="Mean JS divergence")
    plt.title("Behavioral Zone Divergence (Batch Averaged)")
    plt.savefig(os.path.join(out_dir, "c1_zone_divergence.png"), dpi=150)
    plt.close()

def c2_clade_divergence_map(model, tokenizer, assets, sequences, zone_id, delta_months, out_dir):
    """Measures how much the model's predictions change depending on the starting Clade."""
    clade_to_id = assets.get("clade_to_id", {})
    if len(clade_to_id) <= 1:
        return # Not enough clades to compare
        
    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    tokens = [tokenizer.encode(seq, add_special_tokens=True) for seq in sequences]
    batch_tokens = pad_sequences(tokens, maxlen=max_len_input, padding="post", value=1)
    
    # Teacher forcing decoder inputs
    dec_tokens = pad_sequences([t[:assets["MAX_LEN_TARGET"]] for t in tokens], maxlen=assets["MAX_LEN_TARGET"], padding="post", value=tokenizer.pad_token_id)
    dec_input = tf.constant(dec_tokens, dtype="int32")

    clade_probs = {}
    for clade_name, cid in clade_to_id.items():
        t = _build_batch_tensors(assets, tokenizer, batch_tokens, zone_id, cid, delta_months)
        inputs = _route_inputs(model, t, dec_input)
        logits = model.predict(inputs, batch_size=32, verbose=0)
        clade_probs[clade_name] = np.mean(tf.nn.softmax(logits).numpy(), axis=0)

    clades = list(clade_probs.keys())
    n = len(clades)
    div = np.zeros((n, n))
    
    for i, ci in enumerate(clades):
        for j, cj in enumerate(clades):
            p, q = clade_probs[ci], clade_probs[cj]
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log((p + 1e-9) / (m + 1e-9)), axis=-1)
            kl_qm = np.sum(q * np.log((q + 1e-9) / (m + 1e-9)), axis=-1)
            div[i, j] = np.mean(0.5 * kl_pm + 0.5 * kl_qm)

    pd.DataFrame(div, index=clades, columns=clades).to_csv(os.path.join(out_dir, "c2_clade_divergence.csv"))
    
    plt.figure(figsize=(9, 7))
    plt.imshow(div, cmap="plasma")
    plt.xticks(range(n), clades, rotation=90); plt.yticks(range(n), clades)
    plt.colorbar(label="Mean JS divergence")
    plt.title("Evolutionary Path Divergence by Clade")
    plt.savefig(os.path.join(out_dir, "c2_clade_divergence.png"), dpi=150)
    plt.close()

# ============================================================================
# D. Attention & Saliency Introspection
# ============================================================================

def d1_input_attention_saliency(model, tokenizer, assets, sequence, zone_to_id, delta_months, out_dir, default_clade_id=0):
    """
    Computes attention/saliency scores across:
    1. Every individual amino acid position (1 to 550).
    2. Every individual transmission zone.
    3. Every individual clade.
    """
    # Explicitly skip models without positional embeddings (e.g. LSTM)
    if len(model.inputs) != 10:
        print("      -> [SKIPPED D1] Model is an LSTM (does not use positional embeddings).")
        return

    print("      -> Computing comprehensive attention scores for Positions, all Zones, and all Clades...")
    
    max_len_input = assets["MAX_LEN_INPUT_PAD"]
    max_len_target = assets["MAX_LEN_TARGET"]
    clade_to_id = assets.get("clade_to_id", {"Default_Clade": 0})
    
    base_tokens = tokenizer.encode("".join(sequence), add_special_tokens=True)
    batch_tokens = pad_sequences([base_tokens], maxlen=max_len_input, padding="post", value=tokenizer.pad_token_id)
    dec_tokens = pad_sequences([base_tokens[:max_len_target]], maxlen=max_len_target, padding="post", value=tokenizer.pad_token_id)
    dec_input = tf.constant(dec_tokens, dtype="int32")
    
    # 1. Locate Embedding Layers
    embed_layers = {l.name: l for l in model.layers if isinstance(l, tf.keras.layers.Embedding)}
    pos_embed, zone_embed, clade_embed = None, None, None
    
    for name, layer in embed_layers.items():
        name_lower = name.lower()
        if 'zone' in name_lower:
            zone_embed = layer
        elif 'clade' in name_lower:
            clade_embed = layer
        elif 'pos_enc' in name_lower:
            pos_embed = layer

    if not pos_embed or not zone_embed or not clade_embed:
        print(f"      -> [WARNING] Missing embedding layers. Skipping D1.")
        return

    # Helper function to compute saliency gradient for given zone and clade
    def compute_saliency(z_id, c_id):
        t = _build_batch_tensors(assets, tokenizer, batch_tokens, z_id, c_id, delta_months)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(pos_embed.trainable_variables)
            tape.watch(zone_embed.trainable_variables)
            tape.watch(clade_embed.trainable_variables)
            
            inputs = _route_inputs(model, t, dec_input)
            logits = model(inputs, training=False)[0]
            loss = tf.reduce_sum(tf.reduce_max(logits, axis=-1))
            
        e_grad = tape.gradient(loss, pos_embed.trainable_variables[0])
        z_grad = tape.gradient(loss, zone_embed.trainable_variables[0])
        c_grad = tape.gradient(loss, clade_embed.trainable_variables[0])
        
        if isinstance(e_grad, tf.IndexedSlices): e_grad = tf.convert_to_tensor(e_grad)
        if isinstance(z_grad, tf.IndexedSlices): z_grad = tf.convert_to_tensor(z_grad)
        if isinstance(c_grad, tf.IndexedSlices): c_grad = tf.convert_to_tensor(c_grad)
        
        return e_grad, z_grad, c_grad

    # --- A. Positional Saliency ---
    first_zone_id = next(iter(zone_to_id.values())) if zone_to_id else 0
    enc_grad, _, _ = compute_saliency(first_zone_id, default_clade_id)
    
    actual_len = len(sequence)
    
    # -------------------------------------------------------------
    # FIXED: Index enc_grad directly by position p (1 to actual_len)
    # -------------------------------------------------------------
    clean_pos_scores = []
    for pos_idx in range(1, actual_len + 1):
        if enc_grad is not None and pos_idx < enc_grad.shape[0]:
            score = float(np.linalg.norm(enc_grad[pos_idx].numpy()))
        else:
            score = 0.0
        clean_pos_scores.append(score)
    
    pd.DataFrame({"Position": range(1, actual_len + 1), "Attention_Score": clean_pos_scores}).to_csv(
        os.path.join(out_dir, "d1_position_attention.csv"), index=False
    )
    
    plt.figure(figsize=(12, 4))
    plt.plot(range(1, actual_len + 1), clean_pos_scores, color='indigo', linewidth=1.5)
    plt.xlabel("Amino Acid Position (1 to 550)")
    plt.ylabel("Gradient Saliency (Attention)")
    plt.title("Positional Attention Score Across Sequence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "d1_position_attention.png"), dpi=150)
    plt.close()

    # --- B. Per-Zone Attention ---
    zone_records = []
    for zone_name, z_id in zone_to_id.items():
        _, z_grad, _ = compute_saliency(z_id, default_clade_id)
        score = float(np.linalg.norm(z_grad[z_id].numpy())) if z_grad is not None else 0.0
        zone_records.append({"Zone": zone_name, "Attention_Score": score})
        
    df_zones = pd.DataFrame(zone_records).sort_values("Attention_Score", ascending=False)
    df_zones.to_csv(os.path.join(out_dir, "d1_zone_attention.csv"), index=False)
    
    plt.figure(figsize=(10, 5))
    plt.bar(df_zones["Zone"], df_zones["Attention_Score"], color='teal', alpha=0.85)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Saliency / Attention Score")
    plt.title("Attention Scores Across Geographic Zones")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "d1_zone_attention.png"), dpi=150)
    plt.close()

    # --- C. Per-Clade Attention ---
    clade_records = []
    for clade_name, c_id in clade_to_id.items():
        _, _, c_grad = compute_saliency(first_zone_id, c_id)
        score = float(np.linalg.norm(c_grad[c_id].numpy())) if c_grad is not None else 0.0
        clade_records.append({"Clade": clade_name, "Attention_Score": score})
        
    df_clades = pd.DataFrame(clade_records).sort_values("Attention_Score", ascending=False)
    df_clades.to_csv(os.path.join(out_dir, "d1_clade_attention.csv"), index=False)
    
    if len(df_clades) > 1:
        plt.figure(figsize=(10, 5))
        plt.bar(df_clades["Clade"], df_clades["Attention_Score"], color='darkorange', alpha=0.85)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Saliency / Attention Score")
        plt.title("Attention Scores Across Clades")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "d1_clade_attention.png"), dpi=150)
        plt.close()

    print(f"      -> Successfully saved positional, per-zone, and per-clade attention scores.")

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Master directory containing all season folders")
    ap.add_argument("--train_fasta", required=True, help="Master FASTA file")
    ap.add_argument("--out_dir", default="./inspection_results")
    ap.add_argument("--clade_id", type=int, default=0)
    ap.add_argument("--delta_months", type=float, default=10.0)
    args = ap.parse_args()

    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    master_df = load_fasta(args.train_fasta)

    
    # 1. Custom objects dictionary from your working script
    custom_objects_dict = {
        'PositionContextGating': PositionContextGating,
        'TimeDecayLayer': TimeDecayLayer,
        'SpatioTemporalFiLM': SpatioTemporalFiLM,
        'create_padding_mask_logic': create_padding_mask_logic,
        'create_look_ahead_mask_logic': create_look_ahead_mask_logic
    }

    # 2. Dynamic Folder Parsing
    folder_pattern = re.compile(r"(\d{4})_season_\(train_(\d{4})_(\d{4})\)")
    season_folders = sorted([f for f in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, f))])

    for sf in season_folders:
        match = folder_pattern.search(sf)
        if not match:
            continue
            
        target_year, start_year, end_year = map(int, match.groups())
        season_dir = os.path.join(args.base_dir, sf)
        
        print("\n" + "═"*70)
        print(f"🚀 INTROSPECTING SEASON: {target_year}")
        print("═"*70)
        
        # 3. Filter for training window specific to THIS season
        train_start = pd.to_datetime(f"{start_year}-07-01")
        train_end = pd.to_datetime(f"{end_year}-01-31") 
        
        season_df = master_df[(master_df['date'] >= train_start) & (master_df['date'] <= train_end)]

        train_seqs = season_df['sequence'].tolist()

        print(f"   Training DF Duration: {train_start.strftime('%b/%Y')} to {train_end.strftime('%b/%Y')} | Total Sequences: {len(train_seqs)}")

        print("🌍 Extracting global clades from season FASTA...")
        unique_clades = set()
        for header in season_df['id']:
            fields = header.split('|')
            if len(fields) > 2:
                unique_clades.add(fields[2].strip())
                
        sorted_clades = sorted(list(unique_clades))
        global_clade_to_id = {clade: idx for idx, clade in enumerate(sorted_clades)}
        print(f"✅ Found {len(global_clade_to_id)} unique clades.")
        
        
        
        max_samples = 100
        if len(train_seqs) > max_samples:
            train_seqs = np.random.choice(train_seqs, max_samples, replace=False).tolist()
            
        if not train_seqs:
            print(f"⚠️ No training sequences found. Skipping.")
            continue

        # 4. Find models for this season (matching your working glob search)
        model_paths = glob.glob(os.path.join(season_dir, "**", "*.keras"), recursive=True)
        if not model_paths:
            model_paths = glob.glob(os.path.join(season_dir, "**", "*.h5"), recursive=True)

        for weights_path in model_paths:
            model_name = os.path.basename(weights_path)
            if model_name.endswith('_last.keras') or model_name.endswith('_last.h5'):
                continue
                
            variant_name = os.path.basename(os.path.dirname(weights_path))
            print(f"\n  Processing Model: {variant_name} ({model_name})")
            
            assets_path = os.path.join(os.path.dirname(weights_path), "model_assets.pkl")
            if not os.path.exists(assets_path):
                print("   [WARNING] Missing model_assets.pkl. Skipping.")
                continue
            
            # Load assets using your existing loader function
            assets = load_assets(assets_path)

            if "clade_to_id" not in assets or not assets["clade_to_id"]:
                print("   [INFO] 'clade_to_id' missing in assets. Using season-specific temporary mapping for this run.")
                # We create a shallow copy so we don't accidentally mutate global state
                assets = assets.copy() 
                assets["clade_to_id"] = global_clade_to_id
            else:
                print(f"   [INFO] Found pre-existing 'clade_to_id' with {len(assets['clade_to_id'])} clades.")

            zone_to_id = assets["zone_to_id"]
            default_zone_id = next(iter(zone_to_id.values())) if zone_to_id else 0
            
            variant_out_dir = os.path.join(args.out_dir, str(target_year), variant_name)
            os.makedirs(variant_out_dir, exist_ok=True)
            
            # 5. Load model safely with your custom objects
            model = tf.keras.models.load_model(
                weights_path, 
                custom_objects=custom_objects_dict, 
                compile=False, 
                safe_mode=False
            )
            
            #is_transformer = len(model.inputs) == 10

            
            print("   -> Section A: Weight Introspection")
            a1_learned_substitution_geometry(model, tokenizer, variant_out_dir)
            a2_zone_embedding_clustering(model, zone_to_id, variant_out_dir)
            a3_clade_embedding_clustering(model, assets, variant_out_dir)
            a4_contextual_substitution_geometry(
                model=model,
                tokenizer=tokenizer,
                assets=assets,
                train_seqs=train_seqs,
                zone_id=default_zone_id,
                clade_id=args.clade_id,
                out_dir=variant_out_dir
            )

            
            print("   -> Section B: Perturbation Epistasis")
            max_len_input = assets["MAX_LEN_INPUT_PAD"]
            base_sequence = get_consensus(train_seqs, max_len_input - 2)
            positions_to_test = list(range(len(base_sequence))) # Update with actual epitope indices
              
            b1_epistasis_scan_masked_probe(model, tokenizer, assets, base_sequence, train_seqs, zone_to_id, args.clade_id, args.delta_months, positions_to_test, variant_out_dir)

            

            print("   -> Section C: Context Effects")
            c1_zone_divergence_map(model, tokenizer, assets, train_seqs, zone_to_id, args.clade_id, args.delta_months, variant_out_dir)
            c2_clade_divergence_map(model, tokenizer, assets, train_seqs, default_zone_id, args.delta_months, variant_out_dir)
                

            t1_temporal_sensitivity(model, tokenizer, assets, base_sequence, default_zone_id, args.clade_id, variant_out_dir)
            t2_historical_decay_probe(model, tokenizer, assets, base_sequence, default_zone_id, args.clade_id, variant_out_dir)
            print("    -> Section D: Input Attention Extraction")
            

            d1_input_attention_saliency(model, tokenizer, assets, base_sequence, zone_to_id, args.delta_months, variant_out_dir, default_clade_id=args.clade_id)

            
            
            tf.keras.backend.clear_session()

    print(f"\n✅ All outputs written to {args.out_dir}")

if __name__ == "__main__":
    main()