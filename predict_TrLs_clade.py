#!/usr/bin/env python3
"""
=================================================================
Usage:
    python predict_TrLs_clade.py --base_dir h3n2_clean --fasta_file h3n2_clean.fasta
All other parameters can be overridden via flags (see --help).
"""

# ── GPU library fix ───────────────────────────────────────────────────────────
import os, sys, subprocess
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import csv
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley


# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from itertools import chain
from dateutil.relativedelta import relativedelta
import warnings
import math
from typing import Dict, List, Optional, Tuple
import re
import gc
import glob


# ── Numeric / data ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for server use


# ── Deep learning ─────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Transformers / tokenizer ──────────────────────────────────────────────────
from transformers import EsmTokenizer
from models_ym_clade import build_transformer_model


# ── Epitope configuration (HA0-based, matching training) ─────────────────
SP_LENGTH = 16
epitope_sites = {
    'A': [122,124,126,130,131,132,133,135,137,138,140,142,143,144,145,146,150,152,168],
    'B': [128,129,155,156,157,158,159,160,163,164,165,186,187,188,189,190,192,193,194,196,197,198],
    'C': [44,45,46,47,48,50,51,53,54,273,275,276,278,279,280,294,297,299,300,304,305,307,308,309,310,311,312],
    'D': [96,102,103,117,121,167,170,171,172,173,174,175,176,177,179,182,201,203,207,208,209,
            212,213,214,215,216,217,218,219,226,227,228,229,230,238,240,242,244,246,247,248],
    'E': [57,59,62,63,67,75,78,80,81,82,83,86,87,88,91,92,94,109,260,261,262,265],
}
ALL_EPITOPE_POSITIONS_1BASED = sorted(set(chain.from_iterable(epitope_sites.values())))
EPITOPE_INDICES_0BASED       = [p - 1 for p in ALL_EPITOPE_POSITIONS_1BASED]


# ── Project module ────────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════


def parse_eval_args():
    import argparse
    p = argparse.ArgumentParser(description="Rolling Backtest Evaluation Protocol")
    p.add_argument("--base_dir", type=str, required=True, help="Path to the master backtest folder (e.g., backtest_seasonal_h3n2)")
    p.add_argument("--fasta_file", type=str, required=True, help="FASTA file used for data generation")
    p.add_argument("--test_sample_index", type=int, default=0)
    return p.parse_known_args()[0]

def setup_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        return '/GPU:0'
    else:
        return '/CPU:0'
temp = 1.2
ref_anc = 0.8
epi_anch = 4.0

# ══════════════════════════════════════════════════════════════════════════════
# Prediction & Inference Helpers 
# ══════════════════════════════════════════════════════════════════════════════
@tf.function(reduce_retracing=True)
def _predict_step_eager(model, inputs):
    return model(inputs, training=False)

def _build_stable_lock_map(train_df, seq_max_len, tokenizer, lock_threshold=0.95, window_months=6, train_end_date=None, protected_indices=None):
    if train_end_date is None: train_end_date = train_df['date'].max()
    protected_set = set(protected_indices) if protected_indices is not None else set()
    window_start = train_end_date - relativedelta(months=window_months)
    recent_df    = train_df[train_df['date'] >= window_start]
    if recent_df.empty: recent_df = train_df
    raw_seqs    = [list(s.ljust(seq_max_len, '-')) for s in recent_df['sequence'].tolist()]
    seqs_matrix = np.array(raw_seqs, dtype='U1')
    lock_map, lock_freqs = {}, {}
    for pos in range(seq_max_len):
        if pos in protected_set: continue
        col       = seqs_matrix[:, pos]
        mask      = (col != '-') & (col != 'X')
        valid_aas = col[mask]
        if len(valid_aas) == 0: continue
        aas, counts = np.unique(valid_aas, return_counts=True)
        idx         = np.argmax(counts)
        dom_aa      = aas[idx]
        dom_freq    = counts[idx] / len(valid_aas)
        if dom_freq >= lock_threshold:
            toks = tokenizer.encode(dom_aa, add_special_tokens=False)
            if toks:
                lock_map[pos], lock_freqs[pos] = toks[0], dom_freq
    return lock_map, lock_freqs


def _build_global_emergence_map(
    df, seq_max_len, tokenizer, 
    train_end_date, emerge_recent_mo, emerge_boost_scale, emerge_min_delta
):
    global_boosts = {}
    recent_cutoff = train_end_date - pd.DateOffset(months=emerge_recent_mo)
    
    recent_df = df[df['date'] >= recent_cutoff]
    hist_df = df[df['date'] < recent_cutoff]
    
    if recent_df.empty or hist_df.empty:
        return {}

    for pos in range(seq_max_len):
        r_counts = recent_df['sequence'].str[pos].value_counts(normalize=True)
        h_counts = hist_df['sequence'].str[pos].value_counts(normalize=True)
        
        pos_boosts = {}
        for aa, r_freq in r_counts.items():
            h_freq = h_counts.get(aa, 0.0)
            delta = r_freq - h_freq
            
            if delta >= emerge_min_delta:
                t_id = tokenizer.convert_tokens_to_ids(aa)
                if t_id != tokenizer.unk_token_id:
                    pos_boosts[t_id] = delta * emerge_boost_scale
        
        if pos_boosts:
            global_boosts[pos] = pos_boosts
            
    return global_boosts

def _build_global_emergence_map_dynamic(df, seq_len, tokenizer, train_end_date, recent_mo, boost_scale, base_delta, sensitivity_map):
    emergence_map = {}
    t2_end = train_end_date
    t2_start = t2_end - relativedelta(months=recent_mo)
    t1_start = t2_start - relativedelta(months=recent_mo * 2)
    df_past = df[(df['date'] >= t1_start) & (df['date'] < t2_start)]
    df_recent = df[df['date'] >= t2_start]
    if df_past.empty or df_recent.empty: return emergence_map
    for pos in range(seq_len):
        site_specific_threshold = base_delta * sensitivity_map[pos]
        counts_past = df_past['sequence'].str[pos].value_counts(normalize=True)
        counts_recent = df_recent['sequence'].str[pos].value_counts(normalize=True)
        boosts = {}
        for aa, freq_recent in counts_recent.items():
            delta = freq_recent - counts_past.get(aa, 0.0)
            if delta > site_specific_threshold:
                tid = tokenizer.convert_tokens_to_ids(aa)
                if tid is not None:
                    boosts[int(tid)] = float(delta * boost_scale * (1.0 / sensitivity_map[pos]))
        if boosts: emergence_map[pos] = boosts
    return emergence_map

def _build_recent_reference(train_df, target_dt, tokenizer, max_len, recent_months=12):
    cutoff = target_dt - relativedelta(months=recent_months)
    recent_df = train_df[train_df['date'] >= cutoff]
    if len(recent_df) == 0: recent_df = train_df
    seq_max = max(recent_df['sequence'].str.len())
    ref_tokens = [tokenizer.cls_token_id]
    for aa_pos in range(seq_max):
        aas = [s[aa_pos] for s in recent_df['sequence'] if len(s) > aa_pos]
        if not aas:
            ref_tokens.append(tokenizer.pad_token_id)
            continue
        majority_aa, _ = Counter(aas).most_common(1)[0]
        tok = tokenizer.convert_tokens_to_ids(majority_aa)
        ref_tokens.append(tok if tok is not None else tokenizer.pad_token_id)
    while len(ref_tokens) < max_len: ref_tokens.append(tokenizer.pad_token_id)
    return np.array(ref_tokens[:max_len], dtype='int32')







# =============================================================================
# ESSENTIAL HELPERS FOR PRE-COMPUTATION
# =============================================================================

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

def safe_token_to_id(tokenizer, aa: str) -> Optional[int]:
    if aa not in STANDARD_AAS:
        return None
    tid = tokenizer.convert_tokens_to_ids(aa)
    if tid is None:
        return None
    decoded = tokenizer.decode([tid], skip_special_tokens=True).replace(" ", "")
    if decoded != aa:
        return None
    return tid

def build_id_to_aa_map(tokenizer) -> Dict[int, str]:
    mapping = {}
    for aa in STANDARD_AAS:
        tid = safe_token_to_id(tokenizer, aa)
        if tid is not None:
            mapping[tid] = aa
    return mapping

def compute_phylo_weights(dates, train_end_date, half_life_days: float = 180.0) -> np.ndarray:
    lam = math.log(2.0) / half_life_days
    delta = np.array([(train_end_date - d).days for d in dates], dtype="float32")
    delta = np.clip(delta, 0.0, None)
    weights = np.exp(-lam * delta)
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights

def weighted_freq(series_values, weights, normalize: bool = True) -> dict:
    counts: Dict[str, float] = defaultdict(float)
    for aa, w in zip(series_values, weights):
        if aa not in ("-", "X") and aa in STANDARD_AAS:
            counts[aa] += w
    if normalize:
        total = sum(counts.values())
        if total > 0:
            counts = {aa: v / total for aa, v in counts.items()}
    return dict(counts)

def build_epistasis_matrix(train_df_full, seq_max_len: int, phylo_weights: np.ndarray,
                           min_pair_freq: float = 0.02, top_k_partners: int = 8, 
                           min_seq_dist: int = 3) -> Dict[int, List]:
    seqs = train_df_full["sequence"].values
    N = len(seqs)
    weights = phylo_weights[:N]
    aa_to_int = {aa: idx for idx, aa in enumerate(STANDARD_AAS)}
    
    seq_matrix = np.full((N, seq_max_len), -1, dtype=np.int32)
    for row_idx, seq in enumerate(seqs):
        for col_idx, char in enumerate(seq):
            if col_idx < seq_max_len:
                seq_matrix[row_idx, col_idx] = aa_to_int.get(char, -1)

    epistasis_map = {}
    for i in range(1, seq_max_len):
        mi_scores = []
        col_i = seq_matrix[:, i]
        for j in range(seq_max_len):
            # Skip self-comparisons
            if i == j: 
                continue
                
            # Skip immediate neighbors to avoid trivial 1D covariance (e.g., codon constraints)
            if abs(i - j) < min_seq_dist: 
                continue
                
            col_j = seq_matrix[:, j]
            valid_mask = (col_i >= 0) & (col_j >= 0)

            if not np.any(valid_mask): continue
            
            w_valid = weights[valid_mask]
            total_w = w_valid.sum()
            if total_w < 1e-9: continue

            flat_indices = col_i[valid_mask] * 20 + col_j[valid_mask]
            joint_counts = np.bincount(flat_indices, weights=w_valid, minlength=400).reshape(20, 20)
            joint_prob = joint_counts / total_w
            
            marg_i = joint_prob.sum(axis=1)
            marg_j = joint_prob.sum(axis=0)
            denom = np.outer(marg_i, marg_j)
            
            mask = joint_prob >= min_pair_freq
            if not np.any(mask): continue
                
            mi = np.sum(joint_prob[mask] * np.log(joint_prob[mask] / (denom[mask] + 1e-12)))
            if mi > 0.015: mi_scores.append((mi, j, joint_counts))

        mi_scores.sort(key=lambda x: x[0], reverse=True)
        partners = []
        for _, j, joint_counts in mi_scores[:top_k_partners]:
            cond_norm = {}
            marg_j_counts = joint_counts.sum(axis=0)
            for aj_idx in range(20):
                if marg_j_counts[aj_idx] > 0:
                    aj_char = STANDARD_AAS[aj_idx]
                    dist = {}
                    for ai_idx in range(20):
                        val = joint_counts[ai_idx, aj_idx]
                        if val > 0: dist[STANDARD_AAS[ai_idx]] = float(val / marg_j_counts[aj_idx])
                    if dist: cond_norm[aj_char] = dist
            partners.append((j, cond_norm))
        if partners: epistasis_map[i] = partners

    return epistasis_map

def build_empirical_lock_map_with_entropy_guard(
    train_df_full, seq_max_len, aa_to_id, train_end_date, lock_window_mo=3, 
    lock_threshold=0.95, entropy_guard_months=3, max_lock_entropy=0.15
):
    lock_cutoff = train_end_date - timedelta(days=lock_window_mo * 30)
    recent_cutoff = train_end_date - timedelta(days=entropy_guard_months * 30)
    lock_df = train_df_full[train_df_full["date"] >= lock_cutoff]
    recent_df = train_df_full[train_df_full["date"] >= recent_cutoff]
    
    lock_map = {}
    for pos in range(seq_max_len):
        pos_seqs = lock_df[lock_df["sequence"].str.len() > pos]["sequence"].str[pos]
        if pos_seqs.empty: continue
        counts = pos_seqs.value_counts(normalize=True)
        if counts.max() < lock_threshold: continue
        if len(counts) > 1 and counts.iloc[1] >= 0.005: continue
        
        recent_seqs = recent_df[recent_df["sequence"].str.len() > pos]["sequence"].str[pos]
        if not recent_seqs.empty:
            recent_counts = recent_seqs.value_counts(normalize=True).values
            recent_entropy = float(-np.sum(recent_counts * np.log(recent_counts + 1e-12)))
            if recent_entropy > max_lock_entropy: continue 
        
        top_aa = counts.index[0]
        if top_aa in aa_to_id: lock_map[pos] = aa_to_id[top_aa]
    return lock_map


def compute_shift_profile(
    train_df_full, seq_max_len: int, reference_seq: str, phylo_weights: np.ndarray, aa_to_id: dict,
    train_end_date, epitope_set: set, recent_mo: int = 3, emerge_boost_scale: float = 10.0,
    emerge_min_delta: float = 0.005, epitope_boost_factor: float = 1.5, max_boost: float = 5.0,
) -> dict:
    global_boosts = defaultdict(dict)
    recent_cutoff = train_end_date - timedelta(days=recent_mo * 30)
    recent_mask = (train_df_full["date"] >= recent_cutoff).values
    older_mask = ~recent_mask

    w_recent = phylo_weights.copy(); w_recent[~recent_mask] = 0.0
    w_older = phylo_weights.copy(); w_older[~older_mask] = 0.0
    if w_recent.sum() > 0: w_recent /= w_recent.sum()
    if w_older.sum() > 0: w_older /= w_older.sum()

    for pos in range(seq_max_len):
        ref_aa = reference_seq[pos] if pos < len(reference_seq) else "X"
        valid = train_df_full["sequence"].str.len() > pos
        if not valid.any(): continue

        aas = train_df_full.loc[valid, "sequence"].str[pos].values
        freq_recent = weighted_freq(aas, w_recent[valid.values])
        freq_older = weighted_freq(aas, w_older[valid.values])

        for aa, f_new in freq_recent.items():
            if aa == ref_aa or aa not in STANDARD_AAS: continue
            delta = f_new - freq_older.get(aa, 0.0)
            if delta < emerge_min_delta: continue
            boost = min(max_boost, emerge_boost_scale * delta)
            if pos in epitope_set: boost *= epitope_boost_factor
            tid = aa_to_id.get(aa)
            if tid is not None: global_boosts[pos][tid] = boost

    pos_max_boost = np.zeros(seq_max_len + 2, dtype=np.float32)
    for pos, boosts in global_boosts.items():
        if pos + 1 < len(pos_max_boost):
            pos_max_boost[pos + 1] = max(boosts.values()) if boosts else 0.0

    return {"global_boosts": global_boosts, "pos_max_boost": pos_max_boost}


# =============================================================================
# OPTIMIZED PRECOMPUTE FUNCTION FOR V0
# =============================================================================

def precompute_generation_constraints(
    tokenizer, train_df_full, target_date_str, base_year, base_month,
    MAX_LEN_INPUT_PAD, MAX_LEN_TARGET, seq_max_len, reference_seq, epitope_set,
    phylo_half_life_days=180.0, historical_min_freq=0.001, temperature=1.0,
    stable_lock_window_mo=6, stable_lock_threshold=0.95,
    emerge_recent_mo=3, emerge_boost_scale=8.0, emerge_min_delta=0.002,
    epistasis_top_k=5, n_fourier_harmonics=4, entropy_guard_months: int = 3,
    max_lock_entropy: float = 0.15
):
    print("\n--- Pre-computing Shared Data Constraints for Multi-Model Testing ---")

    id_to_aa = build_id_to_aa_map(tokenizer)
    aa_to_id = {aa: tid for tid, aa in id_to_aa.items()}
    train_end_date = train_df_full["date"].max()

    try:
        target_dt = datetime.strptime(target_date_str, "%Y-%m")
    except ValueError:
        target_dt = datetime.now()

    # Phylogenetic weights
    phylo_weights = compute_phylo_weights(
        train_df_full["date"], train_end_date, half_life_days=phylo_half_life_days
    )

    # Richer temporal vector
    d_months = (target_dt.year - base_year) * 12 + (target_dt.month - base_month)
    m_sin = np.sin(2 * np.pi * target_dt.month / 12.0)
    m_cos = np.cos(2 * np.pi * target_dt.month / 12.0)
    time_vec = np.array([float(d_months), float(m_sin), float(m_cos)], dtype="float32")

    # 1. Historical vocabulary mask + pos_volatility (NEW: computed together to avoid double loop)
    allowed_tokens_per_pos = {}
    pos_volatility = {}                                                          # NEW

    for pos in range(seq_max_len):
        mask = train_df_full["sequence"].str.len() > pos
        sub  = train_df_full[mask]
        if sub.empty:
            pos_volatility[pos] = 0.05                                          # NEW
            continue

        w_sub = phylo_weights[mask.values]
        aas   = sub["sequence"].str[pos].values
        freq  = weighted_freq(aas, w_sub / w_sub.sum())
        allowed_tokens_per_pos[pos] = {
            aa_to_id[aa] for aa, f in freq.items()
            if f >= historical_min_freq and aa in aa_to_id
        }

        # Per-position volatility: std of yearly dominant-AA frequency            NEW
        yearly_max = sub.groupby(sub["date"].dt.year)["sequence"].apply(
            lambda x: x.str[pos].value_counts(normalize=True).max()
        )
        pos_volatility[pos] = max(float(yearly_max.std()), 0.01)               # NEW

    # 2. Sensitivity map (NEW) + Dynamic temperature map (existing, kept identical)
    sensitivity_map  = np.ones(MAX_LEN_INPUT_PAD, dtype="float32")             # NEW
    temperature_map  = np.ones(MAX_LEN_INPUT_PAD, dtype="float32") * temperature

    for pos in range(min(seq_max_len, MAX_LEN_INPUT_PAD)):
        hist_std = pos_volatility.get(pos, 0.05)                               # NEW: use cached value

        # sensitivity_map (NEW — mirrors original ensemble Phase 1 logic)
        if hist_std < 0.02:                                                     # NEW
            sensitivity_map[pos] = 0.4 if pos not in epitope_set else 0.6      # NEW
        elif hist_std > 0.10:                                                   # NEW
            sensitivity_map[pos] = 1.3                                         # NEW
        # else: stays 1.0                                                       # NEW

        # temperature_map (existing logic — unchanged)
        if hist_std < 0.02:
            temperature_map[pos] = 0.5
        elif hist_std > 0.10:
            temperature_map[pos] = 1.0
        else:
            temperature_map[pos] = 0.8

    # 3. Empirical lock map (unchanged)
    empirical_lock_map = build_empirical_lock_map_with_entropy_guard(
        train_df_full=train_df_full,
        seq_max_len=seq_max_len,
        aa_to_id=aa_to_id,
        train_end_date=train_end_date,
        lock_window_mo=stable_lock_window_mo,
        lock_threshold=stable_lock_threshold,
        entropy_guard_months=entropy_guard_months,
        max_lock_entropy=max_lock_entropy,
    )

    # 4. Shift-aware emergence profile (existing — extended to also return penalties
    #    and high_velocity_tokens if compute_shift_profile supports them; see note below)
    ref_seq_str = "".join(reference_seq) if isinstance(reference_seq, list) else reference_seq
    shift_pack = compute_shift_profile(
        train_df_full=train_df_full,
        seq_max_len=seq_max_len,
        reference_seq=ref_seq_str,
        phylo_weights=phylo_weights,
        aa_to_id=aa_to_id,
        train_end_date=train_end_date,
        epitope_set=epitope_set,
        recent_mo=emerge_recent_mo,
        emerge_boost_scale=emerge_boost_scale,
        emerge_min_delta=max(emerge_min_delta, 0.005),
    )

    # global_penalties & high_velocity_tokens: pulled from shift_pack if available,
    # otherwise reconstructed from the EMA momentum loop (mirrors ensemble Phase 1).
    if "global_penalties" in shift_pack and "high_velocity_tokens" in shift_pack:
        global_penalties      = shift_pack["global_penalties"]                 # NEW
        high_velocity_tokens  = shift_pack["high_velocity_tokens"]             # NEW
    else:
        # Fallback: recompute from raw monthly EMA (same logic as old ensemble Phase 1)
        global_penalties     = defaultdict(dict)                               # NEW
        high_velocity_tokens = defaultdict(set)                                # NEW

        for pos in range(seq_max_len):
            pos_sequences = train_df_full[train_df_full["sequence"].str.len() > pos].copy()
            if pos_sequences.empty:
                continue

            pos_sequences["month"] = pos_sequences["date"].dt.to_period("M")
            all_months = pd.period_range(
                start=pos_sequences["month"].min(),
                end=pos_sequences["month"].max(), freq="M"
            )
            monthly_counts = pd.crosstab(
                pos_sequences["month"],
                pos_sequences["sequence"].str[pos],
                normalize="index"
            ).reindex(all_months, fill_value=0.0)

            fast_ema = monthly_counts.ewm(span=3,  adjust=False).mean()
            slow_ema = monthly_counts.ewm(span=12, adjust=False).mean()
            latest_momentum = (fast_ema - slow_ema).iloc[-1]

            vol                = pos_volatility.get(pos, 0.05)
            momentum_threshold = 0.02 + (vol * 0.2)

            for aa in latest_momentum.index:
                if aa in ("X", "-"):
                    continue
                token_id = aa_to_id.get(aa)
                if token_id is None:
                    continue

                mom = latest_momentum[aa]
                if mom < -momentum_threshold:
                    global_penalties[pos][token_id] = abs(mom) * 40.0          # NEW
                elif mom > momentum_threshold and mom > 0.04:
                    high_velocity_tokens[pos].add(token_id)                    # NEW

    # pos_max_boost: existing value from shift_pack (already returned), kept as-is
    # but ensure it covers MAX_LEN_TARGET length
    pos_max_boost = shift_pack["pos_max_boost"]                                # existing

    '''
    # 5. Epistasis matrix
    print("Building epistasis co-variation matrix ...")
    epistasis_map = build_epistasis_matrix(
        train_df_full, seq_max_len, phylo_weights, top_k_partners=epistasis_top_k
    )
    '''

    return {
        # ── existing keys (untouched) ─────────────────────────────────────
        "time_vec":               time_vec,
        "target_dt":              target_dt,
        "allowed_tokens_per_pos": allowed_tokens_per_pos,
        "temperature_map":        temperature_map,
        "empirical_lock_map":     empirical_lock_map,
        "global_boosts":          shift_pack["global_boosts"],
        "pos_max_boost":          pos_max_boost,
        # "epistasis_map":        epistasis_map,

        # ── new keys (required by generate_ensemble_guided_consensus) ─────
        "pos_volatility":         pos_volatility,          # NEW
        "sensitivity_map":        sensitivity_map,         # NEW
        "global_penalties":       global_penalties,        # NEW
        "high_velocity_tokens":   high_velocity_tokens,    # NEW
        "seq_max_len":            seq_max_len,             # NEW (convenience passthrough)
        "reference_seq":          reference_seq,           # NEW (convenience passthrough)
        "epitope_set":            epitope_set,             # NEW (convenience passthrough)
    }



def mc_dropout_positional_uncertainty(
    transformer, is_transformer, in_seq, time_gpu, zone_gpu, clade_gpu, prop_gpu, in_epi,
    in_pos, dec_pos_ids, ref_gpu, final_tokens, seq_max_len, MAX_LEN_TARGET,
    num_passes=10
):
    """
    Leak-free EPISTEMIC uncertainty via MC-Dropout.
    """
    num_zones = int(in_seq.shape[0])
    dec_input_np = np.zeros((num_zones, MAX_LEN_TARGET), dtype='int32')
    L = min(len(final_tokens) - 1, MAX_LEN_TARGET, seq_max_len)
    dec_input_np[:, :L] = final_tokens[:-1][:L]
    dec_input_tf = tf.constant(dec_input_np)

    all_pass_probs = []
    for _ in range(num_passes):
        if is_transformer:
            # Added clade_gpu to the model inputs
            model_inputs = [in_seq, time_gpu, zone_gpu, clade_gpu, prop_gpu, in_epi, dec_input_tf, in_pos, dec_pos_ids, ref_gpu]
        else:
            model_inputs = [in_seq, time_gpu, zone_gpu, clade_gpu, prop_gpu, in_epi, dec_input_tf, ref_gpu]
            
        preds = transformer(model_inputs, training=True)  # dropout ACTIVE
        probs = tf.nn.softmax(preds, axis=-1).numpy()
        pooled = np.mean(probs, axis=0)  
        all_pass_probs.append(pooled[:L])

    stacked = np.stack(all_pass_probs, axis=0)  
    mean_probs = np.clip(np.mean(stacked, axis=0), 1e-9, 1.0)
    pos_entropy = -np.sum(mean_probs * np.log(mean_probs), axis=-1)

    chosen_ids = np.argmax(mean_probs, axis=-1)
    chosen_probs_per_pass = np.take_along_axis(stacked, chosen_ids[None, :, None], axis=-1)[..., 0]
    pos_epistemic_var = np.var(chosen_probs_per_pass, axis=0)

    positional_entropy = np.full(seq_max_len, np.nan, dtype='float64')
    positional_var = np.full(seq_max_len, np.nan, dtype='float64')
    positional_entropy[:L] = pos_entropy
    positional_var[:L] = pos_epistemic_var

    return float(np.mean(pos_entropy)), positional_entropy, float(np.mean(pos_epistemic_var)), positional_var


def compute_positional_entropy_from_logits(logits):
    """
    Leak-free confidence signal: Shannon entropy of the model's own
    predictive distribution at each position, computed directly from the
    logits produced during blind autoregressive generation.

    Unlike `calculate_prediction_entropy` / `calculate_teacher_forcing_perplexity`,
    this touches NO ground-truth tokens and NO test_df whatsoever — it reuses
    the same `all_pos_logits` array that the autoregressive decode loop in
    generate_pooled_raw_consensus_no_ref_v1 already produced while generating
    the prediction itself. That makes it valid to use for ensemble weighting
    (it doesn't leak test information into the forecast), and it also means
    it works for genuinely future seasons where no test set exists yet.
    """
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    probs = np.clip(probs, 1e-9, 1.0)
    pos_entropy = -np.sum(probs * np.log(probs), axis=-1)
    global_entropy = float(np.mean(pos_entropy))
    return global_entropy, pos_entropy



# =============================================================================
# MAIN FUNCTION
# =============================================================================


def generate_baseline_consensus(train_df, target_date_str, output_dir, tokenizer):
    """
    Predicts a consensus sequence using linear trend projection per position.
    Vectorized and type-safe implementation.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_dt = datetime.strptime(target_date_str, '%Y-%m')
    target_months_from_base = (target_dt.year * 12 + target_dt.month)
    
    # 1. Prepare Data
    df = train_df.copy()
    df['month_idx'] = df['date'].dt.year * 12 + df['date'].dt.month
    
    # Filter for the last 12 months for trend calculation
    cutoff = target_months_from_base - 12
    df_recent = df[df['month_idx'] >= cutoff].copy()
    
    max_len = df['sequence'].str.len().max()
    predicted_aa_list = []
    
    # 2. Iterate by position
    for pos in range(max_len):
        # Extract sequences long enough for this position
        valid_rows = df_recent[df_recent['sequence'].str.len() > pos].copy()
        
        if valid_rows.empty:
            predicted_aa_list.append('-')
            continue
            
        # Use crosstab to get monthly counts (Rows: month, Cols: Amino Acid)
        # This creates a DataFrame with integer index (month_idx) and clean columns
        monthly_counts = pd.crosstab(valid_rows['month_idx'], valid_rows['sequence'].str[pos], normalize='index')
        
        best_aa = '-'
        max_proj_freq = -1.0
        
        for aa in monthly_counts.columns:
            if aa in ('-', 'X', '<PAD>'): continue
            
            # Explicitly cast to float to prevent numpy/pandas type mismatch
            y = monthly_counts[aa].to_numpy(dtype=float)
            x = monthly_counts.index.to_numpy(dtype=float)
            
            # Linear Trend Projection
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                projected_freq = slope * target_months_from_base + intercept
            else:
                projected_freq = y[-1]
            
            # Clamp projection
            projected_freq = max(0.0, min(1.0, projected_freq))
            
            if projected_freq > max_proj_freq:
                max_proj_freq = projected_freq
                best_aa = aa
        
        predicted_aa_list.append(best_aa)
        
    baseline_seq = "".join(predicted_aa_list)
    
    # 3. Export
    fasta_path = os.path.join(output_dir, f'Baseline_Consensus_{target_date_str}.fasta')
    with open(fasta_path, 'w') as fh:
        fh.write(f'>Baseline_Consensus|{target_date_str}\n{baseline_seq}\n')
        
    return baseline_seq, fasta_path
def generate_two_tiered_mutation_report(
    all_models_logits_dict, 
    tokenizer, 
    reference_seq, 
    test_df,
    output_dir, 
    target_date_str
):
    """
    Two-Tiered Hotspot Report augmented with actual Ground Truth validation 
    from the test sequence dataset. Now also outputs the unweighted ensemble FASTA.
    """
    print("\n--- Generating Validated Two-Tiered Hotspot Report ---")
    os.makedirs(output_dir, exist_ok=True)
    
    seq_max_len = len(reference_seq)
    model_names = list(all_models_logits_dict.keys())
    
    # ==========================================
    # 1. EXTRACT TEST DATA TRUTH
    # ==========================================
    test_seqs = test_df['sequence'].tolist()
    test_counts_dict = {}  
    test_totals_dict = {}  
    test_top_aa = {}       
    test_display_str = {}  

    for pos in range(seq_max_len):
        test_aas = [s[pos] for s in test_seqs if pos < len(s) and s[pos] not in ['-', 'X', '<PAD>']]
        if test_aas:
            counts = Counter(test_aas)
            total_aas = len(test_aas)
            
            test_counts_dict[pos] = counts
            test_totals_dict[pos] = total_aas
            
            top_aa = str(counts.most_common(1)[0][0]).strip().upper()
            test_top_aa[pos] = top_aa
            
            aa_props = []
            for aa, count in counts.most_common():
                if count > 5: # Filter noise
                    prop = int(round((count / total_aas) * 100))
                    aa_props.append(f"{str(aa).strip().upper()}({prop})")
            
            if not aa_props:
                prop = int(round((counts[top_aa] / total_aas) * 100))
                aa_props.append(f"{top_aa}({prop})")
                
            test_display_str[pos] = ", ".join(aa_props)
        else:
            test_counts_dict[pos], test_totals_dict[pos] = {}, 0
            test_top_aa[pos], test_display_str[pos] = None, "N/A"

    # ==========================================
    # 2. EVALUATE PREDICTIONS VS TRUTH
    # ==========================================
    report_data = []
    txt_alerts = []
    
    # NEW: List to store the exact predicted amino acids for the FASTA
    hotspot_ensemble_seq_list = []
    
    for i in range(seq_max_len):
        tok_pos = i + 1
        ref_aa = reference_seq[i]
        
        position_perplexities = []
        position_probs = []
        for model_name in model_names:
            logits = all_models_logits_dict[model_name][i]
            probs = tf.nn.softmax(logits).numpy()
            position_probs.append(probs)
            
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            position_perplexities.append(np.exp(entropy))
            
        avg_perplexity = np.mean(position_perplexities)
        
        # Get Ensemble Prediction
        avg_probs = np.mean(position_probs, axis=0)
        best_token_id = int(np.argmax(avg_probs))
        pred_aa = tokenizer.decode([best_token_id], skip_special_tokens=True).replace(' ', '')
        if not pred_aa: pred_aa = 'X'
        
        # NEW: Append this predicted AA to our FASTA sequence list
        hotspot_ensemble_seq_list.append(pred_aa)
            
        # Ground Truth Validation Logic
        actual_top = test_top_aa.get(i, "N/A")
        actual_dist = test_display_str.get(i, "N/A")
        
        is_actual_shift = (actual_top != ref_aa) and (actual_top not in ["N/A", "X", "-"])
        pred_is_mutation = (pred_aa != ref_aa) and (pred_aa not in ['X', '-', '<PAD>'])
        is_pred_correct = (pred_aa == actual_top)
        
        t_counts = test_counts_dict.get(i, {})
        t_total = test_totals_dict.get(i, 0)
        actual_pred_prop = (t_counts.get(pred_aa, 0) / t_total * 100) if t_total > 0 else 0.0

        # Tier Classification
        category = "Conserved / Stable"
        if avg_perplexity > 10.0:
            category = "Tier 2: Volatile Hotspot"
            if is_actual_shift:
                txt_alerts.append(f"✅ HOTSPOT VALIDATED: Pos {tok_pos} (Ref: {ref_aa}) was volatile (PPL: {avg_perplexity:.1f}) and ACTUALLY SHIFTED to {actual_dist}.")
            else:
                txt_alerts.append(f"⚠️ FALSE ALARM / NOISE: Pos {tok_pos} was volatile (PPL: {avg_perplexity:.1f}) but remained stable {actual_dist}.")
                
        elif pred_is_mutation and avg_perplexity < 5.0:
            category = "Tier 1: Confident Shift"
            if is_pred_correct:
                txt_alerts.append(f"🎯 PERFECT HIT: Pos {tok_pos} predicted to shift to {pred_aa}. It was exactly right! ({actual_dist})")
            else:
                txt_alerts.append(f"❌ MISSED SHIFT: Pos {tok_pos} predicted to shift to {pred_aa}, but reality was {actual_dist}. (Pred captured {actual_pred_prop:.1f}%)")

        # Compile CSV Row
        row = [
            tok_pos, ref_aa, pred_aa, f"{avg_perplexity:.2f}", category,
            actual_top, actual_dist, "YES" if is_actual_shift else "NO",
            "YES" if is_pred_correct else "NO", f"{actual_pred_prop:.1f}%"
        ]
        for ppl in position_perplexities: row.append(f"{ppl:.2f}")
        report_data.append(row)

    # ==========================================
    # 3. EXPORT CSV DATABASE
    # ==========================================
    csv_path = os.path.join(output_dir, f'Validated_Hotspot_Report_{target_date_str}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'Position', 'Ref_AA', 'Ensemble_Pred', 'Avg_Perplexity', 'Classification',
            'Actual_Test_Top_AA', 'Actual_Test_Distribution', 'Did_Actual_Shift?', 
            'Is_Pred_Correct?', 'Pred_Realized_Proportion'
        ] + [f'{m}_PPL' for m in model_names]
        writer.writerow(header)
        writer.writerows(report_data)

    # ==========================================
    # 4. EXPORT TEXT ALERT SUMMARY
    # ==========================================
    txt_path = os.path.join(output_dir, f'Validated_Alerts_{target_date_str}.txt')
    with open(txt_path, 'w') as f:
        f.write(f"VALIDATED TWO-TIERED REPORT | Target Date: {target_date_str}\n")
        f.write("=" * 70 + "\n\n")
        f.write("\n".join(txt_alerts) if txt_alerts else "No major shifts or hotspots detected.")

    # ==========================================
    # 5. NEW: EXPORT HOTSPOT ENSEMBLE FASTA
    # ==========================================
    hotspot_fasta_seq = "".join(hotspot_ensemble_seq_list)
    fasta_path = os.path.join(output_dir, f'Ensemble_Hotspot_Unweighted_{target_date_str}.fasta')
    with open(fasta_path, 'w') as fh:
        fh.write(f'>Ensemble_Hotspot_Unweighted|{target_date_str}\n{hotspot_fasta_seq}\n')

    print(f"Validated Report saved to: {txt_path}")
    print(f"Hotspot Unweighted FASTA saved to: {fasta_path}")
    
    return report_data, csv_path, txt_path, fasta_path

@tf.function(reduce_retracing=True)
def run_inference(transformer, is_transformer, in_seq, time_gpu, zone_gpu, prop_gpu, in_epi, dec_input, in_pos, dec_pos_ids, ref_gpu):
    if is_transformer:
        model_inputs = [in_seq, time_gpu, zone_gpu, prop_gpu, in_epi, dec_input, in_pos, dec_pos_ids, ref_gpu]
    else:
        model_inputs = [in_seq, time_gpu, zone_gpu, prop_gpu, in_epi, dec_input, ref_gpu]
    return transformer(model_inputs, training=False)


    
def generate_pooled_raw_consensus_no_ref_v1(
    transformer, tokenizer, precomp, zone_to_id,
    X_train_padded, X_pos_train_padded, X_epitope_train, X_train_proportions, X_train_clade,
    train_df_full, MAX_LEN_TARGET, MAX_LEN_INPUT_PAD,
    target_date_str, output_dir, historical_consensus,
    ref_recent_months=6, test_index=0, **kwargs
):
    """
    CLEANED v2: Entropy-gated decoding + stability-aware reference blending.
    - Stable positions (low historical variance) decode greedily and lean
      harder on the historical reference, regardless of the model's own
      (possibly miscalibrated) confidence.
    - Variable positions keep nucleus sampling so genuine uncertainty /
      emerging variants are still explored.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_id = os.path.basename(kwargs.get("model_name", "unknown_model")).replace(".h5", "").replace(".keras", "")

    # Inference Hyperparameters
    TEMP = 0.3
    TOP_P = 0.90
    REF_BLEND = 0.2                 # confidence-driven component (kept from v1)
    STABILITY_PULL = 0.5            # NEW: historical-stability-driven component
    STABILITY_ENTROPY_THRESHOLD = 0.35   # NEW: below this pooled entropy -> greedy decode
    GREEDY_DECODE = False           # global override; leave False to use per-position gating below

    # 1. Zone and Token Prep
    zone_ids = list(zone_to_id.values())
    num_zones = len(zone_ids)
    x_id = tokenizer.convert_tokens_to_ids('X')
    vocab_size = int(len(tokenizer))
    reference_seq = historical_consensus
    seq_max_len = len(historical_consensus)
    all_pos_logits = np.zeros((seq_max_len, vocab_size), dtype='float32')

    # 2. Reality Mask (Whitelist) + Per-Position Stability Score
    train_end_date = train_df_full["date"].max()
    recent_cutoff = train_end_date - relativedelta(months=12)
    recent_df = train_df_full[train_df_full["date"] >= recent_cutoff]
    recent_seqs = recent_df["sequence"].tolist()
    strict_allowed_tokens = {}
    pos_stability = np.zeros(seq_max_len, dtype='float32')   # NEW: 0 (volatile) -> 1 (fixed)
    MIN_FREQ_THRESHOLD = 0.05

    for pos in range(seq_max_len):
        aas_at_pos = [s[pos] for s in recent_seqs if pos < len(s) and s[pos] not in ('-', 'X')]
        if not aas_at_pos:
            strict_allowed_tokens[pos] = set()
            continue

        counts = Counter(aas_at_pos)
        total = sum(counts.values())
        dominant_freq = counts.most_common(1)[0][1] / total
        pos_stability[pos] = dominant_freq   # NEW: reuse counts already computed, no extra pass

        valid_aas = [aa for aa, count in counts.items() if (count / total) >= MIN_FREQ_THRESHOLD]
        if valid_aas:
            strict_allowed_tokens[pos] = {
                tokenizer.convert_tokens_to_ids(aa) for aa in valid_aas
                if tokenizer.convert_tokens_to_ids(aa) is not None
            }

    # 3. Tensor Preparation
    time_vec = precomp["time_vec"]
    src_prop = np.array(X_train_proportions)
    raw_prop_vec = np.zeros(MAX_LEN_INPUT_PAD, dtype='float32')
    L = min(len(src_prop), MAX_LEN_INPUT_PAD - 2)
    raw_prop_vec[1:1+L] = src_prop[:L]

    def _rep(arr, batch_size):
        if arr.ndim == 1: return tf.constant(np.repeat(arr[np.newaxis, :], batch_size, axis=0))
        return tf.constant(np.repeat(arr[test_index:test_index+1], batch_size, axis=0))

    in_seq = _rep(X_train_padded, num_zones)
    in_pos = _rep(X_pos_train_padded, num_zones)
    in_epi = _rep(X_epitope_train, num_zones)
    zone_gpu = tf.constant(np.array(zone_ids)[:, np.newaxis], dtype='int32')
    time_gpu = tf.constant(np.repeat(time_vec[np.newaxis, :], num_zones, axis=0))
    prop_gpu = tf.constant(np.repeat(np.stack([raw_prop_vec, raw_prop_vec], axis=-1)[np.newaxis], num_zones, axis=0))
    dec_pos_ids = tf.constant(np.tile(np.arange(MAX_LEN_TARGET), (num_zones, 1)))
    x_suppress_tf = np.where(np.arange(vocab_size) == x_id, -1e9, 0.0).astype('float32') if x_id is not None else np.zeros(vocab_size)

    if X_train_clade is not None and len(X_train_clade) > 0:
        flat_clades = np.array(X_train_clade).flatten()
        vals, counts = np.unique(flat_clades, return_counts=True)
        dominant_clade = int(vals[np.argmax(counts)])
    else:
        dominant_clade = 0

    clade_gpu = tf.constant(np.full((num_zones, 1), dominant_clade, dtype='int32'))

    ref_tokens_single = get_reference_tokens(train_df_full, train_df_full['date'].max(), tokenizer, MAX_LEN_INPUT_PAD)
    ref_gpu = tf.constant(np.repeat(ref_tokens_single[np.newaxis, :], num_zones, axis=0))

    is_transformer = len(transformer.inputs) == 10

    @tf.function(reduce_retracing=True)
    def run_inference(dec_input):
        if is_transformer:
            model_inputs = [
                in_seq, time_gpu, zone_gpu, clade_gpu,
                prop_gpu, in_epi, dec_input,
                in_pos, dec_pos_ids, ref_gpu
            ]
        else:
            model_inputs = [
                in_seq, time_gpu, zone_gpu, clade_gpu,
                prop_gpu, in_epi, dec_input,
                ref_gpu
            ]
        return transformer(model_inputs, training=False)

    # 5. Core Inference Loop
    b0_tokens = [int(tokenizer.cls_token_id)]
    log_data = []

    for i in range(seq_max_len):
        ref_aa = historical_consensus[i] if i < len(historical_consensus) else 'X'
        dec_input_np = np.zeros((num_zones, MAX_LEN_TARGET), dtype='int32')
        dec_input_np[:, :len(b0_tokens)] = b0_tokens

        preds = run_inference(tf.constant(dec_input_np))
        raw_logits = preds[:, len(b0_tokens) - 1, :].numpy()

        if i in strict_allowed_tokens:
            mask = np.full(vocab_size, -1e9, dtype='float32')
            for vid in strict_allowed_tokens[i]: mask[vid] = 0.0
            raw_logits += mask

        zone_probs = tf.nn.softmax(raw_logits).numpy()
        zone_entropy = -np.sum(zone_probs * np.log(zone_probs + 1e-9), axis=-1)
        conf_w = np.exp(-zone_entropy / 0.5)
        pooled_probs = np.average(zone_probs, axis=0, weights=conf_w / np.sum(conf_w))

        # --- Stability-aware reference blending (replaces confidence-only blend) ---
        if i < len(reference_seq):
            ref_id = tokenizer.convert_tokens_to_ids(reference_seq[i])
            if ref_id is not None:
                conf = np.max(pooled_probs)
                confidence_blend = (1.0 - conf) * REF_BLEND
                stability_blend = pos_stability[i] * STABILITY_PULL   # NEW: pulls toward ref
                                                                       # at historically fixed
                                                                       # sites even if the model
                                                                       # is (over)confident elsewhere
                blend_factor = min(0.95, confidence_blend + stability_blend)
                pooled_probs = (1.0 - blend_factor) * pooled_probs
                pooled_probs[ref_id] += blend_factor

        logits = np.log(pooled_probs + 1e-9) / TEMP + x_suppress_tf
        final_probs = tf.nn.softmax(logits).numpy()

        sorted_indices = np.argsort(final_probs)[::-1]
        cumulative_probs = np.cumsum(final_probs[sorted_indices])
        cutoff = np.searchsorted(cumulative_probs, TOP_P) + 1

        mask = np.zeros_like(final_probs)
        mask[sorted_indices[:cutoff]] = 1.0
        sampling_probs = (final_probs * mask) / np.sum(final_probs * mask)

        # --- Entropy-gated decode: greedy at stable/confident positions, ---
        # --- sample only where genuine uncertainty remains ---
        pooled_entropy = -np.sum(final_probs * np.log(final_probs + 1e-9))
        decode_greedy = GREEDY_DECODE or (pooled_entropy < STABILITY_ENTROPY_THRESHOLD)

        if decode_greedy:
            best_token_id = int(np.argmax(sampling_probs))
        else:
            best_token_id = int(np.random.choice(vocab_size, p=sampling_probs))

        b0_tokens.append(best_token_id)
        all_pos_logits[i] = logits

        top_k_ids = sorted_indices[:5]
        csv_row = ["Raw_Consensus", i + 1, ref_aa]
        for pid in top_k_ids:
            if final_probs[pid] >= 0.01:
                aa = tokenizer.decode([int(pid)], skip_special_tokens=True).replace(' ', '')
                csv_row.append(f"{aa} ({final_probs[pid]:.2f})")
        log_data.append(csv_row)

    # 6. Epistemic Uncertainty (MC-Dropout)
    _, mc_pos_var = mc_dropout_positional_uncertainty(
        transformer, is_transformer, in_seq, time_gpu, zone_gpu, clade_gpu, prop_gpu, in_epi,
        in_pos, dec_pos_ids, ref_gpu, b0_tokens, seq_max_len, MAX_LEN_TARGET,
        num_passes=10
    )[2:]

    # 7. Finalize and Export
    final_seq = tokenizer.decode(b0_tokens, skip_special_tokens=True).replace(' ', '')
    fasta_path = os.path.join(output_dir, f'Raw_Consensus_NO_REF_{model_id}_{target_date_str}.fasta')
    with open(fasta_path, 'w') as fh: fh.write(f'>Raw_Consensus_NO_REF|{model_id}|{target_date_str}\n{final_seq}\n')

    return final_seq, fasta_path, all_pos_logits, mc_pos_var

def generate_pooled_raw_consensus_no_ref_dynamic(
    transformer, tokenizer, precomp, zone_to_id,
    X_train_padded, X_pos_train_padded, X_epitope_train, X_train_proportions, X_train_clade,
    train_df_full, MAX_LEN_TARGET, MAX_LEN_INPUT_PAD,
    target_date_str, output_dir, historical_consensus,
    ref_recent_months=12, test_index=0, **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    model_id = os.path.basename(kwargs.get("model_name", "unknown_model")).replace(".h5", "").replace(".keras", "")

    # ---------------- Inference Hyperparameters ----------------
    TEMP_MIN = 0.3
    TEMP_MAX = 1.5
    ENTROPY_K = 1.5
    TOP_P_STABLE = 0.90
    TOP_P_SHIFT = 0.99
    REF_BLEND_MAX = 0.30       # cap on pull toward historical_consensus (stable positions)
    REF_BLEND_MIN = 0.0        # floor on that pull (high-shift positions)
    
    TREND_BLEND_MAX = 0.45     
    
    MIN_OCCURRENCE_COUNT = 10   # noise floor ONLY: exclude AAs seen <10x in the window
    GREEDY_DECODE = False

    # 1. Zone and Token Prep
    zone_ids = list(zone_to_id.values())
    num_zones = len(zone_ids)
    x_id = tokenizer.convert_tokens_to_ids('X')
    vocab_size = int(len(tokenizer))
    reference_seq = historical_consensus
    seq_max_len = len(historical_consensus)
    all_pos_logits = np.zeros((seq_max_len, vocab_size), dtype='float32')

    # 2. Recent-history window for trend fitting
    target_dt = datetime.strptime(target_date_str, '%Y-%m')
    target_months_from_base = target_dt.year * 12 + target_dt.month

    train_end_date = train_df_full["date"].max()
    recent_cutoff = train_end_date - relativedelta(months=ref_recent_months)
    recent_df = train_df_full[train_df_full["date"] >= recent_cutoff].copy()
    recent_df["month_idx"] = recent_df["date"].dt.year * 12 + recent_df["date"].dt.month
    recent_seqs = recent_df["sequence"].tolist()

    # ---------------- 3. Per-position trend + two-tier whitelist ----------------
    allowed_tokens = {}   
    trend_prior = {}      
    shift_score = np.zeros(seq_max_len, dtype='float32')

    for pos in range(seq_max_len):
        aas_at_pos = [s[pos] for s in recent_seqs if pos < len(s) and s[pos] not in ('-', 'X')]
        if not aas_at_pos:
            allowed_tokens[pos] = set()
            continue

        counts = Counter(aas_at_pos)

        allowed_tokens[pos] = {
            tokenizer.convert_tokens_to_ids(aa) for aa, c in counts.items()
            if c >= MIN_OCCURRENCE_COUNT and tokenizer.convert_tokens_to_ids(aa) is not None
        }

        valid_rows = recent_df[recent_df['sequence'].str.len() > pos]
        if valid_rows.empty:
            continue

        monthly_counts = pd.crosstab(
            valid_rows['month_idx'], valid_rows['sequence'].str[pos], normalize='index'
        )

        proj_freqs = {}
        max_pos_slope = 0.0

        for aa in monthly_counts.columns:
            if aa in ('-', 'X', '<PAD>'):
                continue
            y = monthly_counts[aa].to_numpy(dtype=float)
            x = monthly_counts.index.to_numpy(dtype=float)

            if len(x) > 1 and np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                projected = slope * target_months_from_base + intercept
            else:
                slope = 0.0
                projected = y[-1] if len(y) else 0.0

            projected = float(max(0.0, min(1.0, projected)))
            tok_id = tokenizer.convert_tokens_to_ids(aa)
            if tok_id is None:
                continue
            proj_freqs[tok_id] = projected

            if slope > max_pos_slope:
                max_pos_slope = slope

        z = sum(proj_freqs.values())
        trend_prior[pos] = {tid: f / z for tid, f in proj_freqs.items()} if z > 0 else {}

        
        shift_score[pos] = min(1.0, max_pos_slope * 8.0)

    # ---------------- 4. Tensor Preparation ----------------
    time_vec = precomp["time_vec"]
    src_prop = np.array(X_train_proportions)
    raw_prop_vec = np.zeros(MAX_LEN_INPUT_PAD, dtype='float32')
    L = min(len(src_prop), MAX_LEN_INPUT_PAD - 2)
    raw_prop_vec[1:1 + L] = src_prop[:L]

    def _rep(arr, batch_size):
        if arr.ndim == 1:
            return tf.constant(np.repeat(arr[np.newaxis, :], batch_size, axis=0))
        return tf.constant(np.repeat(arr[test_index:test_index + 1], batch_size, axis=0))

    in_seq = _rep(X_train_padded, num_zones)
    in_pos = _rep(X_pos_train_padded, num_zones)
    in_epi = _rep(X_epitope_train, num_zones)
    zone_gpu = tf.constant(np.array(zone_ids)[:, np.newaxis], dtype='int32')
    time_gpu = tf.constant(np.repeat(time_vec[np.newaxis, :], num_zones, axis=0))
    prop_gpu = tf.constant(np.repeat(np.stack([raw_prop_vec, raw_prop_vec], axis=-1)[np.newaxis], num_zones, axis=0))
    dec_pos_ids = tf.constant(np.tile(np.arange(MAX_LEN_TARGET), (num_zones, 1)))
    x_suppress_tf = (np.where(np.arange(vocab_size) == x_id, -1e9, 0.0).astype('float32')
                      if x_id is not None else np.zeros(vocab_size))

    if X_train_clade is not None and len(X_train_clade) > 0:
        flat_clades = np.array(X_train_clade).flatten()
        vals, counts = np.unique(flat_clades, return_counts=True)
        dominant_clade = int(vals[np.argmax(counts)])
    else:
        dominant_clade = 0
    clade_gpu = tf.constant(np.full((num_zones, 1), dominant_clade, dtype='int32'))

    ref_tokens_single = get_reference_tokens(train_df_full, train_df_full['date'].max(), tokenizer, MAX_LEN_INPUT_PAD)
    ref_gpu = tf.constant(np.repeat(ref_tokens_single[np.newaxis, :], num_zones, axis=0))

    is_transformer = len(transformer.inputs) == 10

    @tf.function(reduce_retracing=True)
    def run_inference(dec_input):
        if is_transformer:
            model_inputs = [in_seq, time_gpu, zone_gpu, clade_gpu, prop_gpu, in_epi,
                             dec_input, in_pos, dec_pos_ids, ref_gpu]
        else:
            model_inputs = [in_seq, time_gpu, zone_gpu, clade_gpu, prop_gpu, in_epi,
                             dec_input, ref_gpu]
        return transformer(model_inputs, training=False)

    # ---------------- 5. Core Inference Loop ----------------
    b0_tokens = [int(tokenizer.cls_token_id)]
    log_data = []

    for i in range(seq_max_len):
        ref_aa = historical_consensus[i] if i < len(historical_consensus) else 'X'
        dec_input_np = np.zeros((num_zones, MAX_LEN_TARGET), dtype='int32')
        dec_input_np[:, :len(b0_tokens)] = b0_tokens

        preds = run_inference(tf.constant(dec_input_np))
        raw_logits = preds[:, len(b0_tokens) - 1, :].numpy()

        allowed = allowed_tokens.get(i, set())
        if allowed:
            mask = np.full(vocab_size, -1e9, dtype='float32')
            for vid in allowed:
                mask[vid] = 0.0
            raw_logits = raw_logits + mask

        zone_probs = tf.nn.softmax(raw_logits).numpy()
        entropy = -np.sum(zone_probs * np.log(zone_probs + 1e-9), axis=-1)
        mean_entropy = np.mean(entropy)
        pos_shift = float(shift_score[i]) if i < len(shift_score) else 0.0

        # EDITED: Applied TEMP_MAX cap *after* the shift multiplier to prevent runaway temperature scaling
        base_temp = min(TEMP_MAX, max(TEMP_MIN, TEMP_MIN + (mean_entropy * ENTROPY_K)))
        DYNAMIC_TEMP = min(TEMP_MAX, base_temp * (1.0 + 0.5 * pos_shift))
        
        TOP_P = TOP_P_STABLE + (TOP_P_SHIFT - TOP_P_STABLE) * pos_shift

        conf_w = np.exp(-entropy / 0.5)
        pooled_probs = np.average(zone_probs, axis=0, weights=conf_w / np.sum(conf_w))

        pos_trend = trend_prior.get(i, {})
        if pos_trend:
            trend_vec = np.zeros(vocab_size, dtype='float32')
            for tid, f in pos_trend.items():
                trend_vec[tid] = f
            trend_weight = TREND_BLEND_MAX * pos_shift
            pooled_probs = (1.0 - trend_weight) * pooled_probs + trend_weight * trend_vec
            pooled_probs = pooled_probs / np.sum(pooled_probs)

        if i < len(reference_seq):
            ref_id = tokenizer.convert_tokens_to_ids(reference_seq[i])
            if ref_id is not None:
                conf = np.max(pooled_probs)
                dynamic_ref_blend = REF_BLEND_MIN + (REF_BLEND_MAX - REF_BLEND_MIN) * (1.0 - pos_shift)
                blend_factor = (1.0 - conf) * dynamic_ref_blend
                pooled_probs = (1.0 - blend_factor) * pooled_probs
                pooled_probs[ref_id] += blend_factor

        logits = np.log(pooled_probs + 1e-9) / DYNAMIC_TEMP + x_suppress_tf
        final_probs = tf.nn.softmax(logits).numpy()

        sorted_indices = np.argsort(final_probs)[::-1]
        cumulative_probs = np.cumsum(final_probs[sorted_indices])
        cutoff = np.searchsorted(cumulative_probs, TOP_P) + 1

        mask2 = np.zeros_like(final_probs)
        mask2[sorted_indices[:cutoff]] = 1.0
        sampling_probs = (final_probs * mask2) / np.sum(final_probs * mask2)

        if GREEDY_DECODE:
            best_token_id = int(np.argmax(sampling_probs))
        else:
            best_token_id = int(np.random.choice(vocab_size, p=sampling_probs))

        b0_tokens.append(best_token_id)
        all_pos_logits[i] = logits

        top_k_ids = sorted_indices[:5]
        csv_row = ["Raw_Consensus", i + 1, ref_aa, f"{pos_shift:.2f}"]
        for pid in top_k_ids:
            if final_probs[pid] >= 0.01:
                aa = tokenizer.decode([int(pid)], skip_special_tokens=True).replace(' ', '')
                csv_row.append(f"{aa} ({final_probs[pid]:.2f})")
        log_data.append(csv_row)

    # ---------------- 6. Epistemic Uncertainty (MC-Dropout) ----------------
    _, mc_pos_var = mc_dropout_positional_uncertainty(
        transformer, is_transformer, in_seq, time_gpu, zone_gpu, clade_gpu, prop_gpu, in_epi,
        in_pos, dec_pos_ids, ref_gpu, b0_tokens, seq_max_len, MAX_LEN_TARGET,
        num_passes=10
    )[2:]

    # ---------------- 7. Finalize and Export ----------------
    final_seq = tokenizer.decode(b0_tokens, skip_special_tokens=True).replace(' ', '')
    fasta_path = os.path.join(output_dir, f'Raw_Consensus_NO_REF_{model_id}_{target_date_str}.fasta')
    with open(fasta_path, 'w') as fh:
        fh.write(f'>Raw_Consensus_NO_REF|{model_id}|{target_date_str}\n{final_seq}\n')

    log_path = os.path.join(output_dir, f'position_log_{model_id}_{target_date_str}.csv')
    try:
        with open(log_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(["source", "position", "ref_aa", "shift_score",
                              "top1", "top2", "top3", "top4", "top5"])
            writer.writerows(log_data)
    except Exception:
        pass

    return final_seq, fasta_path, all_pos_logits, mc_pos_var
    

def get_recent_trends_for_inference(prop_lookup, target_zone, target_year_month):
    """
    Extracts the latest slopes for a specific zone into a fast lookup dict.
    Returns: {(position, aa_string): slope}
    """
    recent_df = prop_lookup[
        (prop_lookup['zone'] == target_zone) & 
        (prop_lookup['year_month'] == target_year_month)
    ]
    
    # Create a simplified dict for the ensembler
    recent_trend_dict = {}
    for _, row in recent_df.iterrows():
        pos = row['position']
        aa = row['aa_list']
        slope = row['trend_slope']
        recent_trend_dict[(pos, aa)] = slope
        
    return recent_trend_dict


def generate_pooled_raw_consensus_no_ref_ensemble(
    all_models_logits, tokenizer, historical_consensus, 
    output_dir, target_date_str, recent_trend_dict=None, 
    trend_multiplier=2.0, temperature=1.0, model_id="Trend_Ensemble"):
    """
    ENTROPY + TREND ENSEMBLE
    1. Weighs models by confidence (Entropy).
    2. Applies a multiplier to AAs that are actively trending upward in the population.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_models = {k: v for k, v in all_models_logits.items() if k != "Linear_baseline"}
    sample_logits = np.squeeze(list(valid_models.values())[0])
    seq_len = sample_logits.shape[0]
    vocab_size = sample_logits.shape[1]
    
    all_model_probs = {}
    for m_name, logits in valid_models.items():
        m_logits = np.squeeze(logits)
        exp_logits = np.exp(m_logits - np.max(m_logits, axis=-1, keepdims=True))
        all_model_probs[m_name] = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    ensemble_probs = np.zeros((seq_len, vocab_size), dtype=np.float32)

    best_tokens = []
    log_data = []
    
    for pos in range(seq_len):
        ref_aa = historical_consensus[pos] if pos < len(historical_consensus) else 'X'
        
        pos_probs_list = []
        pos_entropies = []
        
        # 1. Entropy Weighting
        for m_name in valid_models.keys():
            m_probs = all_model_probs[m_name][pos]
            entropy = -np.sum(m_probs * np.log(m_probs + 1e-9))
            
            pos_probs_list.append(m_probs)
            pos_entropies.append(entropy)
            
        raw_weights = np.exp(-np.array(pos_entropies) / temperature)
        normalized_weights = raw_weights / np.sum(raw_weights)
        
        final_prob_dist = np.zeros(vocab_size, dtype=np.float32)
        for i, weight in enumerate(normalized_weights):
            final_prob_dist += pos_probs_list[i] * weight
            
        # --- 2. THE TREND BONUS ---
        if recent_trend_dict:
            for aa_idx in range(vocab_size):
                # Only boost viable candidates (e.g., >1% base probability)
                # This prevents boosting random noise
                if final_prob_dist[aa_idx] > 0.01: 
                    aa_str = tokenizer.decode([int(aa_idx)]).replace(' ', '')
                    slope = recent_trend_dict.get((pos, aa_str), 0.0)
                    
                    if slope > 0:
                        # Multiply probability by (1 + slope_bonus)
                        final_prob_dist[aa_idx] *= (1.0 + (trend_multiplier * slope))
            
            # Re-normalize so probabilities sum perfectly to 1.0 again
            final_prob_dist /= np.sum(final_prob_dist)
            
        # 3. Finalize Selection
        ensemble_probs[pos] = final_prob_dist
        top_3_idx = np.argsort(final_prob_dist)[::-1][:3]
        best_tokens.append(int(top_3_idx[0]))
        
        # Build CSV Log row
        csv_row = ["Trend_Weighted", pos + 1, ref_aa]
        for idx in top_3_idx:
            if final_prob_dist[idx] >= 0.01:
                aa_str = tokenizer.decode([int(idx)]).replace(' ', '')
                csv_row.append(f"{aa_str} ({final_prob_dist[idx]:.2f})")
                
        log_data.append(csv_row)

    ensemble_logits = np.log(ensemble_probs + 1e-9)
    final_seq = tokenizer.decode(best_tokens, skip_special_tokens=True).replace(' ', '')

    fasta_path = os.path.join(output_dir, f'Ensemble_{model_id}_{target_date_str}.fasta')
    with open(fasta_path, 'w') as fh:
        fh.write(f'>Ensemble|{model_id}|{target_date_str}\n{final_seq}\n')

    csv_path = os.path.join(output_dir, f'Raw_Probs_{model_id}_{target_date_str}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        max_cols = max([len(row) - 3 for row in log_data] + [0])
        header = ['Mode', 'Position', 'Reference_AA'] + [f'Prediction_{i+1}' for i in range(max_cols)]
        writer.writerow(header)
        writer.writerows(log_data)

    return final_seq, fasta_path, ensemble_logits

def generate_majority_vote_ensemble(
    all_models_logits, tokenizer, output_dir, target_date_str
):
    """
    Hard Majority Voting Ensemble:
    Each model gets exactly 1 vote per position, completely ignoring how 'confident' they are.
    This prevents an overconfident, poorly-performing model from vetoing the good models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Exclude the linear baseline from the neural ensemble if it's in the dict
    valid_models = [m for m in all_models_logits.keys() if m != "Linear_baseline"]
    
    if not valid_models:
        print(" -> ERROR: No valid models found for ensembling.")
        return None, None

    seq_len = all_models_logits[valid_models[0]].shape[0]
    vocab_size = all_models_logits[valid_models[0]].shape[1]
    
    # 1. Convert logits to probabilities and extract top-1 votes
    model_probs = {}
    model_votes = {}
    
    for m in valid_models:
        logits = all_models_logits[m]
        # Standardize to probabilities
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        model_probs[m] = probs
        # Each model casts exactly 1 vote for its top token ID
        model_votes[m] = np.argmax(probs, axis=-1) 

    final_tokens = []
    
    # 2. Tally votes position by position
    for pos in range(seq_len):
        # Gather the votes from all models for this specific position
        votes_at_pos = [model_votes[m][pos] for m in valid_models]
        vote_counts = Counter(votes_at_pos)
        
        # Find the highest number of votes any token received
        max_votes = max(vote_counts.values())
        
        # Find which token(s) got that max number of votes
        candidates = [token for token, count in vote_counts.items() if count == max_votes]
        
        if len(candidates) == 1:
            # Clear majority! The ensemble agrees.
            final_tokens.append(candidates[0])
        else:
            # TIE-BREAKER: If models are split 1-1-1, we break the tie by 
            # averaging their probabilities, but ONLY for the tied candidates.
            avg_probs = np.mean([model_probs[m][pos] for m in valid_models], axis=0)
            best_tied_token = max(candidates, key=lambda t: avg_probs[t])
            final_tokens.append(best_tied_token)

    # 3. Decode the final sequence
    seq_final = tokenizer.decode(final_tokens, skip_special_tokens=True).replace(' ', '')
    
    # 4. Export the Fasta
    fasta_out = os.path.join(output_dir, f'Ensemble_MajorityVote_{target_date_str}.fasta')
    with open(fasta_out, 'w') as fh: 
        fh.write(f'>Ensemble_MajorityVote|{target_date_str}\n{seq_final}\n')
            
    return seq_final, fasta_out

def generate_unified_ensemble_consensus(
    all_models_logits, entropy_tracker, tokenizer, output_dir, target_date_str, temperature=1.0
):
    """
    PROBABILITY-SPACE Entropy Ensemble:
    Converts all models to a strict [0, 1] probability scale BEFORE applying 
    positional entropy weights. This prevents extreme logits from breaking the math.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Validation & Setup ---
    # Only keep neural models that successfully generated entropy
    valid_models = [
        m for m in all_models_logits.keys() 
        if m != "Linear_baseline" and entropy_tracker.get(m) not in [None, 'N/A']
    ]
    
    if not valid_models:
        print(" -> ERROR: No valid models found with Entropy scores.")
        return None, None

    seq_len = all_models_logits[valid_models[0]].shape[0]
    
    model_probs = []
    model_entropies = []
    
    # --- 2. Extract Bounded Probabilities & Entropies ---
    for m in valid_models:
        # CRITICAL FIX: Convert unbounded logits to strictly bounded [0, 1] probabilities
        logits = all_models_logits[m]
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        model_probs.append(probs)
        
        # Extract positional entropy, replacing NaNs with the global entropy average
        g_entropy, p_entropy_array = entropy_tracker[m]
        clean_p_entropy = np.where(np.isnan(p_entropy_array), g_entropy, p_entropy_array)[:seq_len]
        model_entropies.append(clean_p_entropy)
        
    # Shape: (num_models, seq_len, vocab_size)
    stacked_probs = np.stack(model_probs)
    # Shape: (num_models, seq_len)
    stacked_entropies = np.stack(model_entropies)

    # --- 3. Calculate Normalized Positional Weights ---
    # Lower entropy = higher weight. 
    scaled_entropies = -stacked_entropies / temperature
    
    # Numerical stability shift (prevents np.exp from overflowing)
    max_scaled = np.max(scaled_entropies, axis=0, keepdims=True)
    exp_weights = np.exp(scaled_entropies - max_scaled)
    
    # Normalize so the weights across all models sum to exactly 1.0 at every position
    positional_weights = exp_weights / np.sum(exp_weights, axis=0, keepdims=True)
    
    # Expand dimensions so we can multiply against the vocab_size axis
    expanded_weights = np.expand_dims(positional_weights, axis=-1)

    # --- 4. Probability-Space Fusion ---
    # Multiply the bounded probabilities by the calculated entropy weights
    weighted_probs = stacked_probs * expanded_weights
    
    # Sum the weighted probabilities to get the final ensemble distribution
    ensemble_probs = np.sum(weighted_probs, axis=0)
    
    # Pick the token with the highest final weighted probability
    final_tokens = np.argmax(ensemble_probs, axis=-1)

    # --- 5. Export Sequence ---
    seq_final = tokenizer.decode(final_tokens, skip_special_tokens=True).replace(' ', '')
    
    fasta_out = os.path.join(output_dir, f'Ensemble_EntropyWeighted_v2_{target_date_str}.fasta')
    with open(fasta_out, 'w') as fh: 
        fh.write(f'>Ensemble_EntropyWeighted_v2|{target_date_str}\n{seq_final}\n')
            
    return seq_final, fasta_out

def generate_unified_ensemble_consensus1(
    all_models_logits, entropy_tracker, tokenizer, output_dir, target_date_str, temperature=0.5
):
    """
    Unified Ensemble Inference:
    Uses Dynamic Positional Entropy to weight logit-space model contributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = list(all_models_logits.keys())
    generated_seq_len = all_models_logits[model_names[0]].shape[0] 
    
    # --- 1. Validation & Setup ---
    valid_models = []
    for m in model_names:
        data = entropy_tracker.get(m) 
        if data in ['N/A', None] or (isinstance(data, list) and data[0] is None): 
            continue
        try:
            if not np.isnan(float(data[0])): 
                valid_models.append(m)
        except (ValueError, TypeError, IndexError):
            pass
            
    if not valid_models:
        print(" -> ERROR: No models generated valid Entropy scores.")
        return None, None

    # Stack logits: Shape (num_models, seq_len, vocab_size)
    valid_logits = np.stack([all_models_logits[m] for m in valid_models])
    
    # --- 2. Dynamic Positional Entropy Weighting ---
    valid_pos_entropies = []
    for m in valid_models:
        g_entropy, p_entropy_array = entropy_tracker[m]
        # Clean NaNs by substituting global entropy
        clean_p_entropy = np.where(np.isnan(p_entropy_array), g_entropy, p_entropy_array)[:generated_seq_len]
        valid_pos_entropies.append(clean_p_entropy)
        
    pos_entropy_stacked = np.stack(valid_pos_entropies) 
    
    # Calculate weights: Lower entropy = higher weight
    scaled_entropy = -pos_entropy_stacked / temperature
    max_scaled = np.max(scaled_entropy, axis=0, keepdims=True)
    exp_weights = np.exp(scaled_entropy - max_scaled)
    positional_weights = exp_weights / np.sum(exp_weights, axis=0, keepdims=True)
    expanded_weights = np.expand_dims(positional_weights, axis=-1)

    # --- 3. Logit-Space Fusion ---
    # Weight and sum the raw logits to avoid probability saturation/collapse
    weighted_logits = np.sum(valid_logits * expanded_weights, axis=0)
    
    # Final conversion to probabilities for selection
    final_probs = tf.nn.softmax(weighted_logits / temperature, axis=-1).numpy()
    final_tokens = np.argmax(final_probs, axis=-1)

    # --- 4. Export Sequence ---
    seq_final = tokenizer.decode(final_tokens, skip_special_tokens=True).replace(' ', '')
    
    fasta_out = os.path.join(output_dir, f'Ensemble_Pure_Consensus_{target_date_str}.fasta')
    with open(fasta_out, 'w') as fh: 
        fh.write(f'>Ensemble_Pure_Consensus|{target_date_str}\n{seq_final}\n')
            
    return seq_final, fasta_out


def run_global_seasonal_analysis_batch(train_df, test_df, all_models_seqs, output_dir, analysis_name="batch_analysis", epitope_indices=None):
    def get_window_label(dt):
        if pd.isna(dt): return "Unknown"
        return f"{dt.year-1}Feb-{dt.year}Jan" if dt.month == 1 else f"{dt.year}Feb-{dt.year+1}Jan"

    train_df, test_df = train_df.copy(), test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    train_df['window'] = train_df['date'].apply(get_window_label)
    
    window_time_map = train_df.groupby('window', observed=True)['date'].min().sort_values()
    chronological_order = window_time_map.index.tolist()
    
    epitope_set = set(epitope_indices) if epitope_indices is not None else set()
    seq_max_len = int(train_df['sequence'].str.len().max())

    # =========================================================================
    # 1. STRICT REFERENCE CONSENSUS (Matches evaluation.py exactly)
    # =========================================================================
    # Infer the target_year based on the end of the training timeline
    target_year = train_df['date'].max().year
    ref_start = pd.to_datetime(f"{target_year - 1}-07-01")
    ref_end   = pd.to_datetime(f"{target_year}-01-31")
    
    print(f"\nCalculating strict Reference Consensus ({ref_start.date()} to {ref_end.date()})...")
    reference_df = train_df[(train_df['date'] >= ref_start) & (train_df['date'] <= ref_end)]
    ref_seqs = reference_df['sequence'].tolist()
    
    train_final_aa = {}
    for pos in range(seq_max_len):
        aas = [s[pos] for s in ref_seqs if pos < len(s) and s[pos] not in ['-', 'X', '<PAD>']]
        if aas:
            top_aa = Counter(aas).most_common(1)[0][0]
            train_final_aa[pos] = str(top_aa).strip().upper()
        else:
            train_final_aa[pos] = "-"

    # =========================================================================
    # 2. HISTORICAL WINDOWS (For Excel Details History Log)
    # =========================================================================
    print("Pre-calculating chronological window consensuses...")
    train_pos_data = {pos: {} for pos in range(seq_max_len)}
    
    for window in chronological_order:
        win_seqs = train_df[train_df['window'] == window]['sequence'].tolist()
        for pos in range(seq_max_len):
            aas = [s[pos] for s in win_seqs if pos < len(s) and s[pos] not in ['-', 'X', '<PAD>']]
            if aas:
                counts = Counter(aas)
                top_aa, top_count = counts.most_common(1)[0]
                train_pos_data[pos][window] = f"{top_aa} ({(top_count / len(aas)) * 100:.1f}%, n={len(aas)})"
            else:
                train_pos_data[pos][window] = "-"

    # =========================================================================
    # 3. TEST DATA EXTRACTION
    # =========================================================================
    test_seqs = test_df['sequence'].tolist()
    test_consensus_str = {} 
    test_top_aa = {}        
    test_counts_dict = {}   
    test_totals_dict = {}   
    
    for pos in range(seq_max_len):
        test_aas = [s[pos] for s in test_seqs if pos < len(s) and s[pos] not in ['-', 'X', '<PAD>']]
        if test_aas:
            counts = Counter(test_aas)
            total_aas = len(test_aas)
            
            test_counts_dict[pos] = counts
            test_totals_dict[pos] = total_aas
            
            top_aa = str(counts.most_common(1)[0][0]).strip().upper()
            test_top_aa[pos] = top_aa
            
            aa_props = []
            for aa, count in counts.most_common():
                if count > 5:
                    prop = int(round((count / total_aas) * 100))
                    aa_props.append(f"{str(aa).strip().upper()}({prop})")
            
            if not aa_props:
                prop = int(round((counts[top_aa] / total_aas) * 100))
                aa_props.append(f"{top_aa}({prop})")
                
            test_consensus_str[pos] = ", ".join(aa_props)
        else:
            test_consensus_str[pos] = None
            test_top_aa[pos] = None
            test_counts_dict[pos] = {}
            test_totals_dict[pos] = 0

    # =========================================================================
    # 4. EVALUATION & EXPORT
    # =========================================================================
    summary_rows = []
    detail_rows = []

    for model_name, seqs in all_models_seqs.items():
        print(f"Analyzing {model_name}...")
        for seq_name, seq in seqs.items():
            stats = {'shift': {'total': 0, 'correct': 0, 'labels': []}, 
                     'stable': {'total': 0, 'correct': 0, 'wrongs': []}, 
                     'epitope': {'total': 0, 'correct': 0, 'labels': []}}

            for pos in range(len(seq)):
                p_aa = seq[pos]
                pos_data = {'Model': model_name, 'Scope': seq_name, 'Position': pos + 1, 'Pred_AA': p_aa}
                
                for window in chronological_order:
                    pos_data[window] = train_pos_data.get(pos, {}).get(window, "-")

                t_top_logic_aa = test_top_aa.get(pos)
                t_display_str = test_consensus_str.get(pos)
                
                # We now pull tr_final from the strictly defined 7-month window
                tr_final = train_final_aa.get(pos) 

                if t_top_logic_aa and tr_final and tr_final != '-':
                    p_aa_clean = str(p_aa).strip().upper()
                    
                    is_correct = (p_aa_clean == t_top_logic_aa)
                    is_shifted = (t_top_logic_aa != tr_final)
                    is_epitope = (pos in epitope_set)

                    pos_data['Global_Test_Consensus'] = t_display_str
                    pos_data['Status'] = 'SHIFTED' if is_shifted else 'STABLE'
                    pos_data['Match'] = 'YES' if is_correct else 'NO'

                    # --- LABEL FORMATTING (with proportions for wrongs) ---
                    if is_correct:
                        label = f"{pos + 1}(C)"
                    else:
                        t_counts = test_counts_dict.get(pos, {})
                        t_total = test_totals_dict.get(pos, 0)
                        prop = (t_counts.get(p_aa_clean, 0) / t_total) if t_total > 0 else 0.0
                        label = f"{pos + 1}({prop:.2f})"

                    if is_epitope:
                        stats['epitope']['total'] += 1
                        if is_correct: stats['epitope']['correct'] += 1
                        if is_shifted: stats['epitope']['labels'].append(label)

                    if is_shifted:
                        stats['shift']['total'] += 1
                        stats['shift']['labels'].append(label)
                        if is_correct: stats['shift']['correct'] += 1
                    else:
                        stats['stable']['total'] += 1
                        if is_correct: stats['stable']['correct'] += 1
                        else: stats['stable']['wrongs'].append(label)
                else:
                    pos_data['Global_Test_Consensus'], pos_data['Status'], pos_data['Match'] = "N/A", "N/A", "N/A"

                detail_rows.append(pos_data)

            summary_rows.append({
                'Model': model_name, 'Scope': seq_name, 
                'total_shifts': stats['shift']['total'],
                'shift_accuracy': f"{(stats['shift']['correct'] / stats['shift']['total'] * 100):.2f}%" if stats['shift']['total'] > 0 else "0.00%",
                'shifted_positions': ", ".join(stats['shift']['labels']),
                'epitope_accuracy': f"{(stats['epitope']['correct'] / stats['epitope']['total'] * 100):.2f}%" if stats['epitope']['total'] > 0 else "0.00%",
                'stable_accuracy': f"{(stats['stable']['correct'] / stats['stable']['total'] * 100):.2f}%" if stats['stable']['total'] > 0 else "0.00%",
                'stable_wrong_positions': ", ".join(stats['stable']['wrongs'])
            })

    os.makedirs(output_dir, exist_ok=True)
    excel_filename = os.path.join(output_dir, f"seasonal_12m_{analysis_name}.xlsx")
    
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='SUMMARY', index=False)
        df_details = pd.DataFrame(detail_rows)
        cols = ['Model', 'Scope', 'Position', 'Pred_AA'] + chronological_order + ['Global_Test_Consensus', 'Status', 'Match']
        df_details[cols].to_excel(writer, sheet_name='DETAILS', index=False)

    print(f"\n✅ Global Batch Analysis complete: {excel_filename}")
    return pd.DataFrame(summary_rows)



# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_eval_args()
    device = setup_device()

    ESM_TOKENIZER = "facebook/esm2_t6_8M_UR50D"
    base_dir = args.base_dir
    
    tf.random.set_seed(42)
    np.random.seed(42)

    print("\n" + "="*70)
    print("Influenza Seq2Seq Transformer — Rolling Backtest Inference")
    print("="*70)
    print(f"  Master Directory : {base_dir}")
    print(f"  Device           : {device}")
    print("="*70 + "\n")

    if not os.path.exists(base_dir):
        print(f"[ERROR] Directory {base_dir} does not exist!")
        return

    # Master Perplexity Log for all seasons
    shared_txt_log_path = os.path.join(base_dir, "Global_Backtest_Perplexity.txt")
    with open(shared_txt_log_path, 'w', encoding='utf-8') as f:
        f.write("--- Centralized Perplexity Scores for All Backtest Seasons ---\n")

    print("--- Loading FASTA & Tokenizer (Once for all seasons) ---")
    tokenizer = EsmTokenizer.from_pretrained(ESM_TOKENIZER)
    PAD_ID = tokenizer.pad_token_id
    VOCAB_SIZE = tokenizer.vocab_size

    df = load_fasta(args.fasta_file)

    # -------------------------------------------------------------------
    # DYNAMIC FOLDER PARSING
    # Looks for folders matching: 2018_season_(train_2015_2018)
    # -------------------------------------------------------------------
    folder_pattern = re.compile(r"(\d{4})_season_\(train_(\d{4})_(\d{4})\)")
    
    season_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    
    for sf in season_folders:
        match = folder_pattern.search(sf)
        if not match:
            continue
            
        target_year, start_year, end_year = map(int, match.groups())
        season_dir = os.path.join(base_dir, sf)
        TARGET_DATE_STR = f"{target_year}-11"
        
        print("\n" + "═"*70)
        print(f"🚀 EVALUATING SEASON: {target_year} (Train: {start_year}-{end_year})")
        print("═"*70)
        
        # 1. Rebuild Time Windows exactly as they were in training
        all_windows = _build_window_schedule(start_year, end_year)
        FULL_TRAIN_START = all_windows[0][0]
        FULL_TRAIN_END = all_windows[-1][1]
        START_TEST = datetime(end_year, 8, 1)
        END_TEST = datetime(end_year + 1, 1, 31)

        (train_df_full, test_df,
         X_train_seq_full, X_train_time_full,
         X_train_zone_full, X_test_zone_full,
         zone_to_id, X_train_clade_full, max_clade_dim,
         MAX_DELTA_MONTHS, max_zone_dim,
         total_train_pre_dedup, 
         X_train_proportions_list,
         X_train_trends_list, base_year, base_month) = prepare_data(
            df, FULL_TRAIN_START, FULL_TRAIN_END, START_TEST, END_TEST, tokenizer
        )

        if train_df_full.empty:
            print(f"⚠️ Insufficient training data for {target_year}. Skipping.")
            continue

        train_df_full = train_df_full.reset_index(drop=True)
        
        if test_df is None or test_df.empty:
            print(f"ℹ️ No test data found for {target_year}. Proceeding with Inference-only mode.")
            test_df = pd.DataFrame() # Ensure it's an empty dataframe, not None
        else:
            test_df = test_df.reset_index(drop=True)

        # 2. Tokenize & Compute dimensions dynamically for this specific dataset
        X_train_tokens_full = tokenize_sequences(X_train_seq_full, tokenizer)
        MAX_LEN_INPUT_PAD = max(len(t) for t in X_train_tokens_full) + 2
        MAX_LEN_TARGET = MAX_LEN_INPUT_PAD - 1

        X_train_padded_full = pad_sequences(X_train_tokens_full, maxlen=MAX_LEN_INPUT_PAD, padding='post', value=PAD_ID)
        X_pos_train_full = pad_sequences([list(range(len(t))) for t in X_train_tokens_full], maxlen=MAX_LEN_INPUT_PAD, padding='post', value=0)
        X_epitope_full = build_epitope_mask(X_train_padded_full.shape[0], MAX_LEN_INPUT_PAD)

        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else PAD_ID
        
        Y_train_tokens = [t[1:] + [eos_id] for t in X_train_tokens_full]
        
        # Pad to MAX_LEN_TARGET to match the decoder's expected output shape
        Y_train_aligned = pad_sequences(Y_train_tokens, maxlen=MAX_LEN_TARGET, padding='post', value=PAD_ID)

        # 3. Find models trained specifically for this season
        # Check for both .h5 and .keras extensions
        model_paths = glob.glob(os.path.join(season_dir, "**", "*.keras"), recursive=True)
        if not model_paths:
            model_paths = glob.glob(os.path.join(season_dir, "**", "*.h5"), recursive=True)

        if not model_paths:
            print(f"  [WARNING] No trained models found in {season_dir}! Skipping.")
            continue

        print(f"  Found {len(model_paths)} models. Pre-computing constraints...")

        # 4. Precalculate Baseline & Future Consensus
        def get_window_label(dt):
            if pd.isna(dt): return "Unknown"
            return f"{dt.year-1}Feb-{dt.year}Jan" if dt.month == 1 else f"{dt.year}Feb-{dt.year+1}Jan"

        train_df_full['window'] = train_df_full['date'].apply(get_window_label)
        window_time_map = train_df_full.groupby('window', observed=True)['date'].min().sort_values()
        last_window = window_time_map.index.tolist()[-1]
        
        recent_df = train_df_full[train_df_full['window'] == last_window]
        seq_max_len = int(train_df_full['sequence'].str.len().max())
        
        historical_consensus = []
        for pos in range(seq_max_len):
            pos_seqs = recent_df[recent_df['sequence'].str.len() > pos]['sequence'].str[pos]
            pos_seqs = pos_seqs[~pos_seqs.isin(['-', 'X', '<PAD>'])] 
            if not pos_seqs.empty: historical_consensus.append(pos_seqs.mode()[0])
            else: historical_consensus.append("-") 

        epitope_set = set(int(x) for x in EPITOPE_INDICES_0BASED)

        precomputed_data = precompute_generation_constraints(
            tokenizer=tokenizer, train_df_full=train_df_full, target_date_str=TARGET_DATE_STR,
            base_year=base_year, base_month=base_month, MAX_LEN_INPUT_PAD=MAX_LEN_INPUT_PAD,
            MAX_LEN_TARGET=MAX_LEN_TARGET, seq_max_len=seq_max_len,
            reference_seq=historical_consensus, epitope_set=epitope_set
        )

        #epistasis_map = precomputed_data["epistasis_map"]

        season_entropy_txt_path = os.path.join(season_dir, f"Entropy_Test_{TARGET_DATE_STR}.txt")
        f_entropy = open(season_entropy_txt_path, 'w', encoding='utf-8')
        f_entropy.write(f"Prediction Entropy Results | Season: {target_year}\n")
        f_entropy.write("-" * 60 + "\n")

        loaded_transformers = []
        loaded_model_names = []
        loaded_model_paths = []
        all_models_predictions = {}
        
        custom_objects = {
            'PositionContextGating': PositionContextGating,
            'TimeDecayLayer': TimeDecayLayer,
            'SpatioTemporalFiLM': SpatioTemporalFiLM,
            'create_padding_mask_logic': create_padding_mask_logic,
            'create_look_ahead_mask_logic': create_look_ahead_mask_logic
        }

        print("\n--- Pre-loading all seasonal models for Ensembling ---")
        for weights_path in model_paths:
            model_name = os.path.basename(weights_path)
            if model_name.endswith('_last.keras'):
                continue
            
            print(f"  Loading {model_name}...")
            transformer = tf.keras.models.load_model(weights_path, custom_objects=custom_objects, compile=False, safe_mode=False)
            loaded_transformers.append(transformer)
            loaded_model_names.append(model_name)
            loaded_model_paths.append(weights_path)

        all_models_predictions = {}
        all_models_logits = {}
        #ppl_tracker = {}
        entropy_tracker = {}

        # 1. Calculate baseline ONCE, outside the loop
        print("\nGenerating baseline prediction...")
        baseline_out_dir = os.path.join(season_dir, "Linear_baseline")
        os.makedirs(baseline_out_dir, exist_ok=True)
        base_seq, base_path = generate_baseline_consensus(train_df_full, TARGET_DATE_STR, baseline_out_dir, tokenizer)

        all_models_predictions["Linear_baseline"] = {
                "Prediction": base_seq }

        
        _vocab_size_baseline = int(len(tokenizer))
        linear_logits = np.full((MAX_LEN_TARGET, _vocab_size_baseline), -10.0, dtype='float32')
        for _pos, _aa in enumerate(base_seq):
            if _pos >= MAX_LEN_TARGET:
                break
            _tok_id = tokenizer.convert_tokens_to_ids(_aa)
            if _tok_id is not None:
                linear_logits[_pos, _tok_id] = 10.0
        all_models_logits["LinearTrend_v1"] = linear_logits
        
        entropy_tracker["LinearTrend_v1"] = (0.05, np.full(MAX_LEN_TARGET, 0.05, dtype='float32'))
        
        for transformer, model_name, weights_path in zip(loaded_transformers, loaded_model_names, loaded_model_paths):
            
            # 4. GET THE DIRECTORY DIRECTLY FROM THE TRACKED PATH
            model_dir = os.path.dirname(weights_path) 
            OUTPUT_DIR = os.path.join(model_dir, "predictions")
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            print(f"\n  Processing Standalone Model: {model_name}")

            
            base_primary2, base_fasta2, collected_logits2, mc_pos_var2 = generate_pooled_raw_consensus_no_ref_v1(
                transformer=transformer, tokenizer=tokenizer, precomp=precomputed_data, zone_to_id=zone_to_id, 
                X_train_padded=X_train_padded_full, X_pos_train_padded=X_pos_train_full, X_epitope_train=X_epitope_full, 
                X_train_proportions=X_train_proportions_list[args.test_sample_index], X_train_clade=X_train_clade_full, train_df_full=train_df_full, 
                MAX_LEN_TARGET=MAX_LEN_TARGET, MAX_LEN_INPUT_PAD=MAX_LEN_INPUT_PAD, target_date_str=TARGET_DATE_STR,
                output_dir=OUTPUT_DIR, historical_consensus=historical_consensus#, Y_train_aligned = Y_train_aligned
            )

            all_models_predictions[f"{model_name}_v1"] = {
                "Prediction": base_primary2
            }

            # --- LEAK-FREE ENTROPY: derived from this model's own blind
            # autoregressive generation logits above, no test_df involved ---
            global_entropy, pos_entropy_array = compute_positional_entropy_from_logits(collected_logits2)

            _padded_entropy = np.full(MAX_LEN_TARGET, np.nan, dtype='float64')
            _L = min(len(pos_entropy_array), MAX_LEN_TARGET)
            _padded_entropy[:_L] = pos_entropy_array[:_L]
            pos_entropy_array = _padded_entropy

            # Blend in MC-Dropout self-disagreement (also 100% test-free —
            # computed on the model's own generated sequence above). This
            # catches overconfident-but-fragile positions that a single
            # deterministic softmax's entropy alone would miss.
            MC_UNCERTAINTY_WEIGHT = 5.0  # tune to taste; 0.0 disables the blend
            if mc_pos_var2 is not None and global_entropy is not None:
                safe_mc_var = np.nan_to_num(mc_pos_var2, nan=0.0)
                blended = pos_entropy_array.copy()
                L = min(len(blended), len(safe_mc_var))
                blended[:L] = blended[:L] + MC_UNCERTAINTY_WEIGHT * safe_mc_var[:L]
                pos_entropy_array = blended
                global_entropy = float(np.nanmean(pos_entropy_array))

            if global_entropy is not None:
                entropy_tracker[f"{model_name}_Base"] = (global_entropy, pos_entropy_array)
                entropy_tracker[f"{model_name}_v1"] = (global_entropy, pos_entropy_array)
                print(f"  -> Global Entropy (blind, leak-free, MC-blended): {global_entropy:.4f}")
                f_entropy.write(f"Model: {model_name:<35} | Entropy: {global_entropy:.4f}\n")
            else:
                entropy_tracker[f"{model_name}_Base"] = 'N/A'
                entropy_tracker[f"{model_name}_v1"] = 'N/A'
                print("  -> Entropy: N/A")
                f_entropy.write(f"Model: {model_name:<35} | Entropy: N/A\n")

            all_models_logits[f"{model_name}_v1"] = collected_logits2

        f_entropy.close()

        if len(all_models_logits) > 1:
            report_dir = os.path.join(season_dir, "hotspot_reports")
            
            # Call the reporting function using the fully populated dictionary
            generate_two_tiered_mutation_report(
                all_models_logits_dict=all_models_logits,
                tokenizer=tokenizer,
                reference_seq=historical_consensus,
                test_df=test_df,
                output_dir=report_dir,
                target_date_str=TARGET_DATE_STR
            )
        
        pos_entropy_data = {"Position": np.arange(1, MAX_LEN_TARGET + 1)}
        valid_pos_models = 0

        # 2. Extract the positional arrays for every valid model
        for m_name, data in entropy_tracker.items():
            if data != 'N/A' and data[0] is not None:
                # data[1] contains the positional_entropy_array we calculated
                pos_entropy_data[m_name] = data[1]
                valid_pos_models += 1

        # 3. Save to CSV if we have data
        if valid_pos_models > 0:
            pos_entropy_csv_path = os.path.join(season_dir, f"positional_entropy_{TARGET_DATE_STR}.csv")
            pos_entropy_df = pd.DataFrame(pos_entropy_data)
            
            # Fill NaNs with empty strings or a placeholder so the CSV looks clean
            pos_entropy_df = pos_entropy_df.fillna("") 
            
            pos_entropy_df.to_csv(pos_entropy_csv_path, index=False)
            print(f"  -> Saved positional entropy to {pos_entropy_csv_path}")
        else:
            print("  -> Warning: No valid positional entropy data to save.")


        
        if len(loaded_transformers) > 1:
            print("\n Running Token-Level Cross-Model Ensemble predictions")
            ensemble_dir = os.path.join(season_dir, "ensemble_predictions")

            # Define your core model names (without the _Base or _v1 suffix)
            core_models = ["Transformer_3L", "Transformer_5L", "Transformer_4L", "Lstm_baseline"]#, "LinearTrend"]

            # Loop through both versions to ensemble them completely separately
            #for version in ["Base", "v1"]:
              #  print(f"\n -> Running Ensemble subset: {version}")
                
            # Filter for core models AND ensure it ends with the specific version
            filtered_logits = {
                k: v for k, v in all_models_logits.items() 
                if any(core in k for core in core_models) and k.endswith(f"_v1")
            }
            
            filtered_entropy = {
                k: v for k, v in entropy_tracker.items() 
                if any(core in k for core in core_models) and k.endswith(f"_v1")
            }
            
            print(f" -> Ensembling models: {list(filtered_logits.keys())}")

            
                # 2. Unified Ensemble
            
            ens_seq, fasta_ens = generate_unified_ensemble_consensus(
                all_models_logits=filtered_logits,
                entropy_tracker=filtered_entropy,
                tokenizer=tokenizer,
                output_dir=ensemble_dir,
                target_date_str=f"{TARGET_DATE_STR}",
                temperature=1.0
            )
            
            ens_seq2, fasta_ens2 = generate_majority_vote_ensemble(
                all_models_logits=filtered_logits,
                tokenizer=tokenizer,
                output_dir=ensemble_dir,
                target_date_str=f"{TARGET_DATE_STR}"
            )
            # Save to dictionary using distinct key
            all_models_predictions["Ensemble_Entropy"] = {
                "Prediction": ens_seq
            }
            
            all_models_predictions["Ensemble_Majority"] = {
                "Prediction": ens_seq2
            }
        
        
        # -------------------------------------------------------------------
        # EXPORT PREDICTED MUTATIONS & PROBABILITIES TO CSV
        # -------------------------------------------------------------------
        print(f"\n--- Extracting Predicted Mutations for {target_year} ---")
        mutations_records = []

        # Iterate through models that have stored logits
        for model_name, logits in all_models_logits.items():
            # Ensure logits are 2D (seq_len, vocab_size)
            model_logits = np.squeeze(logits)
            
            # Convert logits to probabilities safely (Softmax)
            exp_logits = np.exp(model_logits - np.max(model_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            # Fetch the final string sequence predicted by this model
            pred_seq = all_models_predictions.get(model_name, {}).get("Prediction", "")
            
            # Iterate through the sequence to compare against the historical consensus
            for pos in range(min(len(pred_seq), len(historical_consensus))):
                ref_aa = historical_consensus[pos]
                pred_aa = pred_seq[pos]
                
                # Skip uninformative padding or gap tokens
                if ref_aa in ['-', 'X', '<PAD>'] or pred_aa in ['-', 'X', '<PAD>']:
                    continue
                    
                # Calculate probabilities for all valid positions first
                pos_probs = probs[pos]
                top_indices = np.argsort(pos_probs)[::-1][:3] 
                
                candidates_str = []
                valid_aa_count = 0
                
                for idx in top_indices:
                    token_prob = pos_probs[idx]
                    # Only include candidates with greater than 1% probability
                    if token_prob > 0.01: 
                        token_str = tokenizer.decode([idx]).replace(' ', '')
                        candidates_str.append(f"{token_str} ({token_prob:.2f})")
                        valid_aa_count += 1
                
                # FLAG TRIGGER: 
                # 1. Top-1 prediction differs from consensus (Mutation)
                # OR 2. At least two AAs have > 1% probability (Uncertainty/Competition)
                if (pred_aa != ref_aa) or (valid_aa_count >= 2):
                    cand_final_str = ", ".join(candidates_str)
                    
                    # Distinguish why this position was flagged
                    event_type = "Mutation" if pred_aa != ref_aa else "Competing AAs"
                    
                    mutations_records.append({
                        "Season": target_year,
                        "Model": model_name,
                        "Position": pos + 1, # Biologists use 1-based indexing
                        "Reference_AA": ref_aa,
                        "Predicted_Candidates": cand_final_str,
                        "Event_Type": event_type
                    })

        # Save to CSV
        if mutations_records:
            mut_df = pd.DataFrame(mutations_records)
            mut_csv_path = os.path.join(season_dir, f"Predicted_Mutations_Probabilities_{TARGET_DATE_STR}.csv")
            mut_df.to_csv(mut_csv_path, index=False)
            print(f"  -> Saved {len(mutations_records)} mutation/uncertainty records to {mut_csv_path}")
        else:
            print("  -> No mutations or competing AAs predicted by the standalone models this season.")
        
        # 6. Run Global Analysis for the Season
        if not test_df.empty and len(test_df) > 0:
            print(f"\n--- Running Shift Analysis for Season {target_year} ---")
            global_analysis_df = run_global_seasonal_analysis_batch(
                train_df        = train_df_full,
                test_df         = test_df,
                all_models_seqs = all_models_predictions,
                output_dir      = season_dir,
                analysis_name   = f"all_models_{TARGET_DATE_STR}",
                epitope_indices = EPITOPE_INDICES_0BASED
            )
        else:
            print(f"\n⚠️ Test data missing or empty for {target_year}. Skipping Global Shift Analysis.")
        
        # 7. Comprehensive Memory Cleanup at the end of the Season
        print("\nCleaning up seasonal VRAM allocations...")
        tf.keras.backend.clear_session()
        del loaded_transformers, train_df_full, test_df, X_train_padded_full
        gc.collect()

    print("\n✅ All Backtest Seasons Evaluated Successfully!")

if __name__ == "__main__":
    main()