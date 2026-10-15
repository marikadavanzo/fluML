#!/usr/bin/env python3
"""
Sequence-to-Sequence Model for Influenza Prediction with influenza transmission Zones
v2: Fixed/variable position detection + per-position context injection + Continuous Time

Usage: python models_ym_clade.py <fasta_file> <start_target_year> <end_target_year>
Example: python models_ym_clade.py h3n2_clean.fasta --start_target_year 2014 --end_target_year 2019
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from collections import defaultdict, Counter
import matplotlib.ticker as mticker
import math
from tqdm.notebook import tqdm
import gc

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

from transformers import EsmTokenizer
from Bio import SeqIO

from dateutil.relativedelta import relativedelta
import pickle

# ============================================================================
# BIOLOGY-INFORMED UTILITIES
# ============================================================================

def compute_position_entropy(sequences):
    """
    Compute per-position amino acid entropy from training sequences.
    Higher entropy = more variable = higher learning weight.
    """
    max_len = max(len(s) for s in sequences)
    entropy = np.zeros(max_len)

    for i in range(max_len):
        aas = [s[i] for s in sequences if i < len(s)]
        if not aas:
            continue
        counts = Counter(aas)
        probs = np.array(list(counts.values()), dtype=np.float32)
        probs /= probs.sum()
        entropy[i] = -np.sum(probs * np.log(probs + 1e-8))

    entropy /= entropy.max() + 1e-8
    return entropy  # raw [0,1] — used as variability signal

def get_entropy_summary(entropy_array):
    # Filter out absolute zero (perfectly static sites) to see the "signal"
    active_signals = entropy_array[entropy_array > 0]
    
    if len(active_signals) == 0:
        print("Warning: Absolute zero entropy across all sites.")
        return 0.1 # Fallback
    
    stats = {
        "max": np.max(active_signals),
        "mean": np.mean(active_signals),
        "p50 (median)": np.percentile(active_signals, 50),
        "p75": np.percentile(active_signals, 75),
        "p90": np.percentile(active_signals, 90),
        "p95": np.percentile(active_signals, 95)
    }
    
    print("\n--- Entropy Distribution Summary ---")
    for k, v in stats.items():
        print(f"{k:>12}: {v:.4f}")
    
    # Logic: Target the top 10% of most variable sites
    dynamic_threshold = np.percentile(active_signals, 90)
    print(f"Suggested Threshold (Top 10%): {dynamic_threshold:.4f}")
    
    return dynamic_threshold

# ============================================================================
# FIXED / VARIABLE POSITION ANALYSIS
# ============================================================================

MIN_SEQS_PER_ZONE_WINDOW = 20   # zones with fewer seqs in a window are excluded
FIXED_THRESHOLD = 0.98          # AA must be ≥ threshold % in a zone-window to count as "fixed"
FIXED_LOSS_WEIGHT = 0.3        
VARIABLE_LOSS_BOOST = 5.0       # multiplier on top of entropy for variable positions


def _window_boundaries(year, half):
    if half == "FebJul":
        return datetime(year, 2, 1), datetime(year, 7, 31)
    else:
        return datetime(year, 8, 1), datetime(year + 1, 1, 31)


def _build_window_schedule1(start_year, end_train_year):
    windows, year = [], start_year
    while True:
        for half in ("FebJul", "AugJan"):
            s, e = _window_boundaries(year, half)
            windows.append((s, e))
            if half == "AugJan" and year + 1 == end_train_year:
                return windows
        year += 1

def _build_window_schedule(start_year, end_train_year, end_train_month=None):
    windows, year = [], start_year
    while True:
        for half in ("FebJul", "AugJan"):
            s, e = _window_boundaries(year, half)
            windows.append((s, e))

            # DEFAULT BEHAVIOR: If no month is provided (or if they explicitly pass 1 for Jan)
            if end_train_month is None or end_train_month == 1:
                if half == "AugJan" and year + 1 == end_train_year:
                    return windows
            
            # CUSTOM MONTH BEHAVIOR: Stop at the correct semester block
            else:
                # If month is between Feb-Jul (2 to 7), stop at the FebJul window of the target year
                if year == end_train_year and half == "FebJul" and 2 <= end_train_month <= 7:
                    # Note: 'e' will still be the end of July. If you need 'e' to be exactly 
                    # the month they entered (e.g., June), you must overwrite 'e' here.
                    return windows
                
                # If month is between Aug-Dec (8 to 12), stop at the AugJan window of the target year
                elif year == end_train_year and half == "AugJan" and 8 <= end_train_month <= 12:
                    return windows

        year += 1

def _six_month_window_label(date):
    """Return 'FebJul-YYYY' or 'AugJan-YYYY' for a datetime."""
    m = date.month
    if 2 <= m <= 7:
        return f"FebJul-{date.year}"
    elif m >= 8:
        return f"AugJan-{date.year}"
    else:  # Jan
        return f"AugJan-{date.year - 1}"


def compute_fixed_positions(train_df, max_len, fixed_threshold=FIXED_THRESHOLD,
                             min_seqs=MIN_SEQS_PER_ZONE_WINDOW, protected_indices=None):
    """
    For every (zone, 6-month-window) group in train_df, compute dominant-AA
    frequency at each position.  A position is FIXED if, in EVERY qualifying
    group (≥ min_seqs sequences), the dominant AA frequency ≥ fixed_threshold.
    """
    df = train_df.copy()
    df['window'] = df['date'].apply(_six_month_window_label)

    dominant_freqs = defaultdict(list)

    groups = df.groupby(['zone', 'window'])
    for (zone, window), grp in groups:
        if len(grp) < min_seqs:
            continue
        seqs = grp['sequence'].tolist()
        for pos in range(max_len):
            aas = [s[pos] for s in seqs if pos < len(s) and s[pos] not in ('-', 'X')]
            if len(aas) < 5:
                continue
            counts = Counter(aas)
            dominant_freq = counts.most_common(1)[0][1] / len(aas)
            dominant_freqs[pos].append(dominant_freq)

    protected_set = set(protected_indices) if protected_indices is not None else set()
    fixed_mask = np.zeros(max_len, dtype=bool)
    stats_rows = []
    for pos in range(max_len):
        freqs = dominant_freqs.get(pos, [])
        if not freqs:
            fixed_mask[pos] = False
            stats_rows.append({'position': pos + 1, 'n_groups': 0,
                                'min_dominant_freq': np.nan, 'is_fixed': False})
            continue
        min_dom = min(freqs)
        is_fixed = (min_dom >= fixed_threshold)
        fixed_mask[pos] = is_fixed
        stats_rows.append({'position': pos + 1, 'n_groups': len(freqs),
                            'min_dominant_freq': round(min_dom, 4),
                            'is_fixed': is_fixed})

    stats_df = pd.DataFrame(stats_rows)
    n_fixed = fixed_mask.sum()
    print(f"Position analysis: {n_fixed}/{max_len} fixed "
          f"({100*n_fixed/max_len:.1f}%), "
          f"{max_len - n_fixed} variable.")
    return fixed_mask, stats_df


def build_position_loss_weights(fixed_mask, entropy_signal):
    """
    Combine fixed/variable mask with entropy to produce final per-position
    loss weights for the decoder targets.
    """
    max_len = len(fixed_mask)
    ent = np.ones(max_len, dtype=np.float32)
    n = min(max_len, len(entropy_signal))
    ent[:n] = entropy_signal[:n].astype(np.float32)

    weights = np.where(
        fixed_mask,
        FIXED_LOSS_WEIGHT,
        1.0 + VARIABLE_LOSS_BOOST * ent
    ).astype(np.float32)
    return weights


def generate_aa_summary_csv(train_df, test_df, output_path):
    def get_counts(df):
        if df is None or df.empty:
            return {}
        pos_map = {}
        max_len = df['sequence'].str.len().max()
        for i in range(max_len):
            aas = df['sequence'].str.get(i).dropna()
            if not aas.empty:
                counts = Counter(aas)
                pos_map[i + 1] = ", ".join([f"{c}{aa}" for aa, c in counts.items()])
        return pos_map

    train_counts = get_counts(train_df)
    test_counts  = get_counts(test_df)
    all_pos = sorted(set(train_counts) | set(test_counts))
    rows = [{'position': p, 'AA_traindf': train_counts.get(p, ''),
             'AA_testdf': test_counts.get(p, '')} for p in all_pos]
    pd.DataFrame(rows).to_csv(output_path, index=False)

    if test_df is None or test_df.empty:
        print(f"AA summary CSV (Train Only): {output_path}")
    else:
        print(f"AA summary CSV: {output_path}")



# ============================================================================
# CONFIGURATION
# ============================================================================

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 3
DFF = 256
DROPOUT_RATE = 0.3
D_TIME_EMBED = D_MODEL
D_ZONE_EMBED = 32

MODEL_NAME = 'facebook/esm2_t12_35M_UR50D'

EPOCHS = 50
BATCH_SIZE = 64

TEST_SAMPLE_INDEX = 0

# ============================================================================
# EPITOPE CONFIGURATION (HA0-BASED)
# ============================================================================

SP_LENGTH = 16

epitope_sites = {
    'A': [122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144, 145, 146, 150, 152, 168],
    'B': [128, 129, 155, 156, 157, 158, 159, 160, 163, 164, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198],
    'C': [44, 45, 46, 47, 48, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312],
    'D': [96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248],
    'E': [57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265]
}

ALL_EPITOPE_POSITIONS_1BASED = sorted(set(chain.from_iterable(epitope_sites.values())))
EPITOPE_INDICES_0BASED = [p - 1 for p in ALL_EPITOPE_POSITIONS_1BASED]

print(f"Epitope indices (0-based, HA0): {EPITOPE_INDICES_0BASED}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_epitope_mask(num_sequences, max_len):
    mask = np.zeros((num_sequences, max_len), dtype=np.int32)
    for pos in EPITOPE_INDICES_0BASED:
        if pos < max_len:
            mask[:, pos] = 1
    return mask


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='ESM2 t12_35M fine-tune seq2seq for influenza prediction')
    
    # Original required positional arguments
    parser.add_argument('fasta_file',       type=str)
    parser.add_argument("--start_target_year", type=int, default=2018, 
                        help="First target season year to evaluate")
    parser.add_argument("--end_target_year", type=int, default=2025, 
                        help="Last target season year to evaluate")

    # NEW: Optional manual month overrides
    parser.add_argument('--start-month',    type=int, default=None, help='Manual start month (1-12)')
    parser.add_argument('--end-month',      type=int, default=None, help='Manual end month (1-12)')
    
    parser.add_argument('--epochs',          type=int,  default=EPOCHS)
    parser.add_argument('--batch-size',      type=int,  default=BATCH_SIZE)
    return parser.parse_args()


def load_fasta(fasta_path):
    print(f"Loading FASTA: {fasta_path}...")
    all_parsed_data = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        fields = record.id.split('|')
        if len(fields) < 6:
            continue
        unique_id = fields[0].strip()
        subtype   = fields[1].strip()
        clade     = fields[2].strip()
        date_str  = fields[3].strip()
        country   = fields[4].strip()
        zone      = fields[5].strip()
        try:
            date     = datetime.strptime(date_str, '%Y-%m-%d')
            # 1. Get the raw sequence and remove gaps/stops
            #raw_seq = str(record.seq).replace('-', '').replace('*', '')
            
            # Keep gaps (-), only remove stop codons (*)
            raw_seq = str(record.seq).replace('*', '')
            
        # 2. Check length and cut conditionally
            if len(raw_seq) >= 553:
                sequence = raw_seq[16:]
            else:
                sequence = raw_seq
                
            all_parsed_data.append(dict(
                date=date, sequence=sequence,
                id_full_header=record.id, unique_id=unique_id,
                clade=clade, subtype=subtype, country=country, zone=zone
            ))
        except ValueError:
            continue
    df = pd.DataFrame(all_parsed_data)
    print(f"Total records: {len(df)}")
    return df

def prepare_data(df, start_train, end_train, start_test, end_test, tokenizer):
    print("Preparing data with positional proportions...")
    train_df = df[(df['date'] >= start_train) & (df['date'] <= end_train)].sort_values('date').copy()
    test_df  = df[(df['date'] >= start_test)  & (df['date'] <= end_test)].sort_values('date').copy()

    # 1. Create Time Buckets for historical proportions 
    train_df['year_month'] = train_df['date'].dt.to_period('M')

    # 2. Calculate Positional Proportions
    print("Calculating positional frequencies...")
    exploded = train_df[['year_month', 'zone', 'sequence']].copy()
    exploded['aa_list'] = exploded['sequence'].apply(list)
    exploded = exploded.explode('aa_list')
    exploded['position'] = exploded.groupby(level=0).cumcount()

    group_cols = ['year_month', 'zone', 'position']
    counts = exploded.groupby(group_cols + ['aa_list']).size().reset_index(name='pos_seq_count')
    totals = exploded.groupby(group_cols).size().reset_index(name='pos_total_count')

    prop_lookup = counts.merge(totals, on=group_cols)
    prop_lookup['pos_proportion'] = prop_lookup['pos_seq_count'] / prop_lookup['pos_total_count']
    prop_lookup['ym_num'] = prop_lookup['year_month'].astype(int)
    prop_lookup = prop_lookup.sort_values(['zone', 'position', 'aa_list', 'ym_num'])

    # 3. Compute slope ONCE per (zone, position, aa_list) group
    print("Calculating slopes (vectorized)...")
    slope_group_cols = ['zone', 'position', 'aa_list']

    def fast_slope(g, k=4):
        g = g.sort_values('ym_num').tail(k)
        if len(g) < 2:
            return 0.0
        x = g['ym_num'].values.astype(float)
        y = g['pos_proportion'].values.astype(float)
        x = x - x.mean()
        y = y - y.mean()
        denom = (x ** 2).sum()
        return 0.0 if denom == 0 else float((x * y).sum() / denom)

    slopes = (
        prop_lookup
        .groupby(slope_group_cols)
        .apply(fast_slope, include_groups=False)
        .reset_index(name='trend_slope')
    )

    prop_lookup = prop_lookup.merge(slopes, on=slope_group_cols, how='left')

    # 4. Build lookup dicts for fast sequence mapping
    prop_dict = prop_lookup.set_index(['year_month', 'zone', 'position', 'aa_list'])['pos_proportion'].to_dict()
    trend_dict = prop_lookup.set_index(['year_month', 'zone', 'position', 'aa_list'])['trend_slope'].to_dict()

    def map_seq_to_props_and_trends(row):
        ym = row['year_month']
        z  = row['zone']
        seq = row['sequence']
        props  = []
        trends = []
        for i, aa in enumerate(seq):
            props.append(prop_dict.get((ym, z, i, aa), 0.0))
            trends.append(trend_dict.get((ym, z, i, aa), 0.0))
        return props, trends

    print("Mapping proportions to sequences...")
    train_df[['pos_prop_vector', 'pos_trend_vector']] = train_df.apply(
        map_seq_to_props_and_trends, axis=1, result_type='expand'
    )

    # 5. Standard Processing (Dedup & ID Mapping)
    total_train_pre_dedup = len(train_df)
    train_df = train_df.drop_duplicates(subset=['year_month', 'zone', 'subtype', 'sequence'], keep='first')

    X_train_sequences   = train_df['sequence'].tolist()
    X_train_dates       = train_df['date'].tolist()
    X_train_proportions = train_df['pos_prop_vector'].tolist()
    X_train_trends_list = train_df['pos_trend_vector'].tolist()

    # --- NEW: Continuous Time Vectors ---
    base_year = start_train.year
    base_month = start_train.month

    def get_time_vector(date):
        d_months = (date.year - base_year) * 12 + (date.month - base_month)
        m_sin = np.sin(2 * np.pi * date.month / 12.0)
        m_cos = np.cos(2 * np.pi * date.month / 12.0)
        return [float(d_months), float(m_sin), float(m_cos)]

    X_train_time_list = [get_time_vector(d) for d in X_train_dates]
    X_train_time_input = np.array(X_train_time_list, dtype=np.float32)
    MAX_DELTA_MONTHS = float(np.max(X_train_time_input[:, 0]))

    unique_zones = sorted(list(set(df['zone'])))
    zone_to_id = {z: i for i, z in enumerate(unique_zones)}
    X_train_zone_input = np.array([zone_to_id[z] for z in train_df['zone']]).reshape(-1, 1)
    X_test_zone_input  = np.array([zone_to_id[z] for z in test_df['zone']]).reshape(-1, 1)
    max_zone_dim = len(unique_zones)

    unique_clades = sorted(list(set(df['clade'])))
    clade_to_id = {c: i for i, c in enumerate(unique_clades)}
    X_train_clade_input = np.array([clade_to_id.get(c, 0) for c in train_df['clade']]).reshape(-1, 1)
    max_clade_dim = len(unique_clades)

    return (train_df, test_df, X_train_sequences, X_train_time_input,
            X_train_zone_input, X_test_zone_input,
            zone_to_id, X_train_clade_input, max_clade_dim,
            MAX_DELTA_MONTHS, max_zone_dim, total_train_pre_dedup, 
            X_train_proportions, X_train_trends_list, base_year, base_month)



def tokenize_sequences(sequences, tokenizer):
    def esm_tokenize(seq, tok):
        enc = tok(seq, add_special_tokens=True, return_tensors='np')
        return enc['input_ids'].squeeze().tolist()
    return [esm_tokenize(s, tokenizer) for s in sequences]


def create_padding_mask_logic(seq, pad_id):
    seq = tf.cast(tf.math.equal(seq, pad_id), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask_logic(seq_tensor):
    size = tf.shape(seq_tensor)[1]
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]


class TimeDecayLayer(layers.Layer):
    def __init__(self, max_delta_months=120.0, decay_rate=0.06, learnable=False, **kwargs):
        self.max_delta_months = max_delta_months
        self.init_decay_rate = decay_rate
        self.learnable = learnable
        super(TimeDecayLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.learnable:
            self.decay_rate_weight = self.add_weight(
                name="decay_rate_weight",
                shape=(),
                initializer=tf.keras.initializers.Constant(self.init_decay_rate),
                trainable=True,
            )
        else:
            self.decay_rate_weight = tf.constant(self.init_decay_rate, dtype=tf.float32)

    def call(self, tensors):
        time_input, embedding = tensors
        # time_input shape is (batch, 3). Index 0 holds the delta_months.
        delta_months = time_input[:, 0]
        
        time_diff = self.max_delta_months - delta_months
        # Ensure time_diff is >= 0 so future target dates don't get negative age (exploding weights)
        time_diff = tf.maximum(0.0, time_diff)
        
        decay_factor = tf.exp(-self.decay_rate_weight * time_diff)
        return embedding * tf.expand_dims(decay_factor, -1)

    def get_config(self):
        config = super(TimeDecayLayer, self).get_config()
        config.update({
            "max_delta_months": self.max_delta_months,
            "decay_rate": self.init_decay_rate,
            "learnable": self.learnable,
        })
        return config


def apply_sequence_masking(token_sequences, mask_id, vocab_size, mask_prob=0.15):
    """
    Randomly masks tokens in the encoder input.
    - 80% of masked tokens become <MASK>
    - 10% become a random token
    - 10% stay the same (to keep representations grounded)
    """
    masked_seqs = np.array(token_sequences, copy=True)
    
    rand = np.random.rand(*masked_seqs.shape)
    
    # We only mask "real" amino acids, not padding (ID 1) or CLS (ID 0)
    mask_condition = (rand < mask_prob) & (masked_seqs > 2)
    
    # 80% of the time, replace with [MASK]
    mask_indices = mask_condition & (np.random.rand(*masked_seqs.shape) < 0.8)
    masked_seqs[mask_indices] = mask_id
    
    # 10% of the time, replace with random AA (assume AA tokens are IDs 4-25)
    random_indices = mask_condition & (np.random.rand(*masked_seqs.shape) < 0.5)
    masked_seqs[random_indices] = np.random.randint(4, 25, size=np.count_nonzero(random_indices))
    
    return masked_seqs


def get_reference_tokens(df, end_date, tokenizer, max_len):
    """Computes global consensus of the LAST 6 MONTHS of the training window."""
    start_date = end_date - relativedelta(months=6)
    ref_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if ref_df.empty:  # Fallback to all data if window is empty
        ref_df = df

    raw_seqs = ref_df['sequence'].tolist()
    seq_max = max(len(s) for s in raw_seqs)

    consensus_seq = ""
    for i in range(seq_max):
        aas = [s[i] for s in raw_seqs if i < len(s) and s[i] not in ('-', 'X')]
        consensus_seq += Counter(aas).most_common(1)[0][0] if aas else 'X'

    tokens = tokenizer.encode(consensus_seq, add_special_tokens=True)
    return pad_sequences([tokens], maxlen=max_len, padding='post', value=1)[0]


class LinearWarmup(tf.keras.callbacks.Callback):
    def __init__(self, target_lr, warmup_epochs):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            new_lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            #tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            self.model.optimizer.learning_rate.assign(new_lr)
            print(f"\n - Epoch {epoch+1}: Warmup LR to {new_lr:.6f}")


# ============================================================================
# PER-POSITION CONTEXT GATING LAYER
# ============================================================================
class PositionContextGating(layers.Layer):
    def __init__(self, d_model, d_ctx, gate_bias_init=-0.5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ctx = d_ctx
        self.gate_bias_init = gate_bias_init
        self.ctx_proj = layers.Dense(d_model, use_bias=True)
        self.gate_proj = layers.Dense(
            d_model,
            activation='sigmoid',
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(gate_bias_init),
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, seq_repr, context_vec, training=False):
        ctx = self.ctx_proj(context_vec)  # (batch, 1, d_model)
        seq_len = tf.shape(seq_repr)[1]
        ctx = tf.broadcast_to(ctx, [tf.shape(seq_repr)[0], seq_len, self.d_model])

        gate_input = tf.concat([seq_repr, ctx], axis=-1)
        gate = self.gate_proj(gate_input)  # (batch, seq_len, d_model)

        out = seq_repr + gate * ctx
        return self.layer_norm(out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'd_model': self.d_model,
            'd_ctx': self.d_ctx,
            'gate_bias_init': float(self.gate_bias_init),
        })
        return cfg

@tf.function
def make_padding_mask(seq, pad_id=1):
    mask = tf.math.not_equal(seq, pad_id)
    return mask[:, tf.newaxis, tf.newaxis, :] 

@tf.function
def make_look_ahead_mask(seq_len):
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :] 

@tf.function
def make_combined_mask(decoder_input, pad_id=1):
    seq_len = tf.shape(decoder_input)[1]
    pad = make_padding_mask(decoder_input, pad_id)
    ahead = make_look_ahead_mask(seq_len)
    return tf.logical_and(pad, ahead)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SpatioTemporalFiLM(layers.Layer):
    """
    Feature-wise Linear Modulation (FiLM).
    Instead of simple gating, this scales (gamma) and shifts (beta) 
    the sequence based on the Zone/Time context.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.gamma_dense = layers.Dense(d_model, activation='sigmoid', name='film_gamma')
        self.beta_dense = layers.Dense(d_model, name='film_beta')

    def build(self, input_shape):
        self.gamma_dense.build(input_shape)
        self.beta_dense.build(input_shape)
        
        # Mark the layer as built to satisfy the Keras loader
        super().build(input_shape)

    def call(self, x, context):
        # Reverted to your original signature to maintain backward compatibility 
        # with your saved model graph!
        gamma = self.gamma_dense(context) # (batch, 1, d_model)
        beta = self.beta_dense(context)   # (batch, 1, d_model)
        return x * gamma + beta


def build_transformer_model(num_layers, d_model, num_heads, dff,
                            input_vocab_size, output_vocab_size,
                            max_len_input, max_len_target, rate,
                            max_delta_months, d_time_embed,
                            max_zone_dim, d_zone_embed, max_clade_dim):

    # --- 1. Inputs ---
    encoder_input = layers.Input(shape=(max_len_input,), dtype='int32', name='encoder_input')
    epitope_input = layers.Input(shape=(max_len_input,), dtype='int32', name='epitope_input')
    decoder_input = layers.Input(shape=(max_len_target,), dtype='int32', name='decoder_input')

    encoder_clade_input = layers.Input(shape=(1,), dtype='int32', name='encoder_clade_input')
    
    # NEW: Continuous Time Vector Input (delta_months, month_sin, month_cos)
    encoder_time_input = layers.Input(shape=(3,), dtype='float32', name='encoder_time_input')
    
    encoder_zone_input = layers.Input(shape=(1,), dtype='int32', name='encoder_zone_input')
    position_input_enc = layers.Input(shape=(max_len_input,), dtype='int32', name='position_input_enc')
    position_input_dec = layers.Input(shape=(max_len_target,), dtype='int32', name='position_input_dec')
    proportion_input = layers.Input(shape=(max_len_input, 2), dtype='float32', name='proportion_input')
    reference_input = layers.Input(shape=(max_len_input,), dtype='int32', name='reference_input')

    # --- 2. Spatio-Temporal Context ---
    # Continuous projection instead of categorical embedding
    time_embed = layers.Dense(d_time_embed, activation='gelu', name="time_proj")(encoder_time_input)
    
    # Applying decay to prioritize recent mutations based on continuous age
    time_embed_decayed = TimeDecayLayer(
        max_delta_months=max_delta_months, decay_rate=0.06
    )([encoder_time_input, time_embed])

    zone_embed = layers.Embedding(max_zone_dim, d_zone_embed, name="zone_embed")(encoder_zone_input)
    
    # MLP to learn the interaction/lag between Zone and Time
    # Need to flatten zone_embed from (batch, 1, d_zone) to match time_embed_decayed (batch, d_time)
    zone_embed_flat = layers.Reshape((d_zone_embed,))(zone_embed)
    clade_embed = layers.Embedding(max_clade_dim, d_zone_embed, name="clade_embed")(encoder_clade_input)
    clade_embed_flat = layers.Reshape((d_zone_embed,))(clade_embed)

    # ---> MODIFIED: Update Concatenation <---
    st_interact = layers.Concatenate(axis=-1)([time_embed_decayed, zone_embed_flat, clade_embed_flat])
    
    st_interact = layers.Dense(d_model // 2, activation='gelu')(st_interact)
    st_context = layers.Dense(d_model, activation='gelu', name="st_context_vector")(st_interact)
    st_context = layers.GaussianNoise(0.05)(st_context)
    
    # Reshape back to (batch, 1, d_model) for sequence operations
    st_context_seq = layers.Reshape((1, d_model))(st_context)

    # --- 3. Encoder Architecture ---
    pos_embedding_enc = layers.Embedding(600, d_model, name="pos_enc")
    epitope_embed = layers.Embedding(2, d_model, name="epi_enc")(epitope_input)
    
    # Sequence Branch
    enc_embed = layers.Embedding(input_vocab_size, d_model)(encoder_input)
    prop_proj = layers.Dense(d_model)(layers.Dense(16, activation='gelu')(proportion_input))
    
    enc_output = (enc_embed * tf.math.sqrt(tf.cast(d_model, tf.float32)) + 
                  pos_embedding_enc(position_input_enc) + epitope_embed + prop_proj)
    
    # Apply FiLM Gating
    film_layer = SpatioTemporalFiLM(d_model)
    enc_output = film_layer(enc_output, st_context_seq)

    # Reference Branch (Baseline)
    ref_embed = layers.Embedding(input_vocab_size, d_model)(reference_input)
    ref_output = film_layer(ref_embed, st_context_seq)

    # Fusion of History and Reference
    fusion_gate = layers.Dense(d_model, activation='sigmoid')(layers.Concatenate()([enc_output, ref_output]))
    combined_memory = enc_output + fusion_gate * ref_output
    combined_memory = layers.LayerNormalization(epsilon=1e-6)(combined_memory)

    # Transformer Blocks
    enc_padding_mask = layers.Lambda(create_padding_mask_logic, arguments={'pad_id': 1})(encoder_input)
    for i in range(num_layers):
        attn = layers.MultiHeadAttention(num_heads, d_model//num_heads, dropout=rate)(
            combined_memory, combined_memory, combined_memory, attention_mask=enc_padding_mask)
        combined_memory = layers.LayerNormalization()(combined_memory + layers.Dropout(rate)(attn))
        
        ffn = layers.Dense(dff, activation='gelu')(combined_memory)
        ffn = layers.Dense(d_model)(ffn)
        combined_memory = layers.LayerNormalization()(combined_memory + layers.Dropout(rate)(ffn))

    # --- 4. Decoder Architecture ---
    pos_embedding_dec = layers.Embedding(600, d_model, name="pos_dec")
    dec_embed = layers.Embedding(output_vocab_size, d_model)(decoder_input)
    dec_output = (dec_embed * tf.math.sqrt(tf.cast(d_model, tf.float32)) + 
                  pos_embedding_dec(position_input_dec))
    
    # Gating the Decoder for regional awareness
    dec_output = film_layer(dec_output, st_context_seq)

    # Masks
    look_ahead_mask = layers.Lambda(create_look_ahead_mask_logic)(decoder_input)
    dec_pad_mask = layers.Lambda(create_padding_mask_logic, arguments={'pad_id': 1})(decoder_input)
    combined_mask = layers.Maximum()([dec_pad_mask, look_ahead_mask])

    for i in range(num_layers):
        # Self-Attention
        attn1 = layers.MultiHeadAttention(num_heads, d_model//num_heads, dropout=rate)(
            dec_output, dec_output, dec_output, attention_mask=combined_mask)
        dec_output = layers.LayerNormalization()(dec_output + layers.Dropout(rate)(attn1))

        # Cross-Attention to "Traveling" Context
        '''
        ctx_attn = layers.MultiHeadAttention(num_heads, d_model//num_heads, dropout=rate)(
            dec_output, st_context_seq, st_context_seq)
        dec_output = layers.LayerNormalization()(dec_output + layers.Dropout(rate)(ctx_attn))
        '''

        # Inject Spatio-Temporal Context at each layer
        dec_output = film_layer(dec_output, st_context_seq)

        # Cross-Attention to Historical Memory
        attn2 = layers.MultiHeadAttention(num_heads, d_model//num_heads, dropout=rate)(
            dec_output, combined_memory, combined_memory, attention_mask=enc_padding_mask)
        dec_output = layers.LayerNormalization()(dec_output + layers.Dropout(rate)(attn2))

        # FFN
        ffn_dec = layers.Dense(dff, activation='gelu')(dec_output)
        ffn_dec = layers.Dense(d_model)(ffn_dec)
        dec_output = layers.LayerNormalization()(dec_output + layers.Dropout(rate)(ffn_dec))

    # Output
    #final_output = layers.Dense(output_vocab_size, name='final_output')(dec_output)
    # 1. Generate the standard Neural Network logits
    nn_logits = layers.Dense(
        output_vocab_size, 
        kernel_initializer='zeros', 
        bias_initializer='zeros', 
        name='nn_logits'
    )(dec_output)
    
    # 2. Build the Skip Connection for the Linear Trend
    # Real per-vocab projection (was: log of a scalar broadcast-added to every class — a no-op after softmax)
    trend_logits = layers.Dense(
        output_vocab_size,
        kernel_initializer='zeros',   # start neutral: nn_logits alone drives early training
        bias_initializer='zeros',
        name='trend_residual_proj'
    )(proportion_input)
    
    # 3. Align the sequence lengths (CRITICAL FIX)
    trend_logits_aligned = layers.Lambda(
        lambda x: x[:, 1:, :],
        name='align_trend_shift'
    )(trend_logits)
    
    # 4. Add them together!
    final_output = layers.Add(name='final_output')([nn_logits, trend_logits_aligned])
    
    return Model(
        inputs=[
            encoder_input, encoder_time_input, encoder_zone_input, encoder_clade_input,
            proportion_input, epitope_input, decoder_input,
            position_input_enc, position_input_dec, reference_input
        ],
        outputs=final_output
    )


def build_LSTM_model(num_layers, d_model, dff,
                            input_vocab_size, output_vocab_size,
                            max_len_input, max_len_target, rate,
                            max_delta_months, d_time_embed,
                            max_zone_dim, d_zone_embed, max_clade_dim): # <-- Added max_clade_dim

    # --- 1. Inputs ---
    encoder_input = layers.Input(shape=(max_len_input,), dtype='int32', name='encoder_input')
    epitope_input = layers.Input(shape=(max_len_input,), dtype='int32', name='epitope_input')
    decoder_input = layers.Input(shape=(max_len_target,), dtype='int32', name='decoder_input')
    
    # NEW: Continuous Time Vector Input (delta_months, month_sin, month_cos)
    encoder_time_input = layers.Input(shape=(3,), dtype='float32', name='encoder_time_input')
    
    encoder_zone_input = layers.Input(shape=(1,), dtype='int32', name='encoder_zone_input')
    encoder_clade_input = layers.Input(shape=(1,), dtype='int32', name='encoder_clade_input') # <-- Added clade input
    
    # Note: position_input_enc and position_input_dec are removed from inputs for LSTM
    proportion_input = layers.Input(shape=(max_len_input, 2), dtype='float32', name='proportion_input')
    reference_input = layers.Input(shape=(max_len_input,), dtype='int32', name='reference_input')

    # --- 2. Spatio-Temporal Context ---
    # Continuous projection instead of categorical embedding
    time_embed = layers.Dense(d_time_embed, activation='gelu', name="time_proj")(encoder_time_input)
    
    # Applying decay to prioritize recent mutations based on continuous age
    time_embed_decayed = TimeDecayLayer(
        max_delta_months=max_delta_months, decay_rate=0.06
    )([encoder_time_input, time_embed])

    zone_embed = layers.Embedding(max_zone_dim, d_zone_embed, name="zone_embed")(encoder_zone_input)
    
    # MLP to learn the interaction/lag between Zone, Time, and Clade
    zone_embed_flat = layers.Reshape((d_zone_embed,))(zone_embed)
    
    # <-- Added clade embedding logic
    clade_embed = layers.Embedding(max_clade_dim, d_zone_embed, name="clade_embed")(encoder_clade_input)
    clade_embed_flat = layers.Reshape((d_zone_embed,))(clade_embed)

    # <-- Updated concatenation to include clade
    st_interact = layers.Concatenate(axis=-1)([time_embed_decayed, zone_embed_flat, clade_embed_flat])
    
    st_interact = layers.Dense(d_model // 2, activation='gelu')(st_interact)
    st_context = layers.Dense(d_model, activation='gelu', name="st_context_vector")(st_interact)
    st_context = layers.GaussianNoise(0.05)(st_context)
    
    # Reshape back to (batch, 1, d_model) for sequence operations
    st_context_seq = layers.Reshape((1, d_model))(st_context)

    # --- 3. Encoder Architecture (LSTM) ---
    # Positional embeddings are removed because LSTMs natively handle sequence order.
    epitope_embed = layers.Embedding(2, d_model, name="epi_enc")(epitope_input)
    enc_embed = layers.Embedding(input_vocab_size, d_model)(encoder_input)
    prop_proj = layers.Dense(d_model)(layers.Dense(16, activation='gelu')(proportion_input))
    
    # Combine sequence embeddings
    enc_output = enc_embed * tf.math.sqrt(tf.cast(d_model, tf.float32)) + epitope_embed + prop_proj
    
    # Apply FiLM Gating using your Spatio-Temporal context
    film_layer = SpatioTemporalFiLM(d_model)
    enc_output = film_layer(enc_output, st_context_seq)

    # Reference Branch (Baseline)
    ref_embed = layers.Embedding(input_vocab_size, d_model)(reference_input)
    ref_output = film_layer(ref_embed, st_context_seq)

    # Fusion of History and Reference
    fusion_gate = layers.Dense(d_model, activation='sigmoid')(layers.Concatenate()([enc_output, ref_output]))
    combined_memory = enc_output + fusion_gate * ref_output
    combined_memory = layers.LayerNormalization(epsilon=1e-6)(combined_memory)

    # Native Keras Boolean Mask (Assuming 1 is your pad_id)
    enc_mask = layers.Lambda(lambda x: tf.cast(tf.math.not_equal(x, 1), tf.bool))(encoder_input)

    # LSTM Encoder
    # Using Bidirectional to capture past and future context. 
    # d_model // 2 ensures the concatenated forward/backward states equal d_model.
    encoder_lstm = layers.Bidirectional(
        layers.LSTM(d_model // 2, return_sequences=True, return_state=True)
    )
    enc_seq, forward_h, forward_c, backward_h, backward_c = encoder_lstm(combined_memory, mask=enc_mask)
    
    # Concatenate hidden and cell states to pass to the decoder
    state_h = layers.Concatenate()([forward_h, backward_h])
    state_c = layers.Concatenate()([forward_c, backward_c])


    # --- 4. Decoder Architecture (LSTM) ---
    dec_embed = layers.Embedding(output_vocab_size, d_model)(decoder_input)
    dec_output = dec_embed * tf.math.sqrt(tf.cast(d_model, tf.float32))
    
    # Gating the Decoder embeddings for regional awareness
    dec_output = film_layer(dec_output, st_context_seq)

    dec_mask = layers.Lambda(lambda x: tf.cast(tf.math.not_equal(x, 1), tf.bool))(decoder_input)

    # LSTM Decoder
    # Initialize with the final states from the Encoder
    decoder_lstm = layers.LSTM(d_model, return_sequences=True, return_state=True)
    dec_seq, _, _ = decoder_lstm(dec_output, initial_state=[state_h, state_c], mask=dec_mask)

    # Inject Spatio-Temporal Context into the sequence output
    dec_seq = film_layer(dec_seq, st_context_seq)

    # Cross-Attention to Historical Memory 
    # (Highly recommended in LSTMs to match Transformer performance on longer sequences)
    attention_layer = layers.Attention()
    context_vector = attention_layer([dec_seq, enc_seq], mask=[dec_mask, enc_mask])
    
    dec_combined = layers.Concatenate()([dec_seq, context_vector])
    
    # FFN Equivalent
    ffn_dec = layers.Dense(dff, activation='gelu')(dec_combined)
    dec_combined = layers.LayerNormalization()(dec_combined + layers.Dropout(rate)(ffn_dec))

    # Final Output
    nn_logits = layers.Dense(
        output_vocab_size, 
        kernel_initializer='zeros', 
        bias_initializer='zeros', 
        name='nn_logits'
    )(dec_combined)
    
    # 2. Project the linear trend data to match the vocabulary size
    trend_logits = layers.Dense(output_vocab_size, name='trend_residual_proj')(proportion_input)
    def apply_slope_mask(inputs):
        logits, props = inputs
        mask = tf.where(tf.abs(props[:, :, 1:]) > 0.05, 1.0, 0.0)
        return logits * tf.cast(mask, logits.dtype)

    trend_logits = layers.Lambda(apply_slope_mask, name='slope_mask_lambda')([trend_logits, proportion_input])
    
    # 3. Align the sequence lengths (Shift by 1 to skip the [CLS] padding)
    trend_logits_aligned = layers.Lambda(
        lambda x: x[:, 1:, :], 
        name='align_trend_shift'
    )(trend_logits)
    
    # 4. Add them together! The NN acts as a non-linear corrector.
    final_output = layers.Add(name='final_output')([nn_logits, trend_logits_aligned])

    return Model(
        inputs=[
            encoder_input, encoder_time_input, encoder_zone_input, encoder_clade_input, # <-- Added encoder_clade_input here
            proportion_input, epitope_input, decoder_input,
            reference_input  
        ],
        outputs=final_output
    )

# ============================================================================
# CUSTOM METRICS
# ============================================================================

class MaskedSparseCategoricalAccuracy(Metric):
    def __init__(self, pad_id, name='masked_sparse_categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_id        = pad_id
        self.total_correct = self.add_weight(name='total_correct', shape=(), initializer='zeros', dtype=tf.float32)
        self.total_samples = self.add_weight(name='total_samples', shape=(), initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true       = tf.cast(y_true, tf.int64)
        y_pred_tokens= tf.argmax(y_pred, axis=-1, output_type=tf.int64)
        mask         = tf.cast(tf.math.not_equal(y_true, self.pad_id), self.dtype)
        matches      = tf.cast(tf.math.equal(y_true, y_pred_tokens), self.dtype) * mask
        self.total_correct.assign_add(tf.reduce_sum(matches))
        self.total_samples.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total_correct / (self.total_samples + K.epsilon())

    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_samples.assign(0.0)

def weighted_masked_loss(y_true, y_pred, pad_id, position_weights, label_smoothing=0.1):
    vocab_size = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=vocab_size)
    
    pw = tf.cast(position_weights, tf.float32)
    seq_len = tf.shape(y_true)[1]
    padded_pw = tf.pad(pw, [[0, tf.maximum(0, seq_len - tf.shape(pw)[0])]], constant_values=1.0)
    weights = padded_pw[:seq_len] 
    
    is_variable = tf.cast(weights > 0.02, tf.float32) 
    is_variable = tf.expand_dims(tf.expand_dims(is_variable, axis=0), axis=-1)
    
    smooth_positives = 1.0 - label_smoothing
    smooth_negatives = label_smoothing / tf.cast(vocab_size - 1, tf.float32)
    smoothed_labels = y_true_one_hot * smooth_positives + (1.0 - y_true_one_hot) * smooth_negatives
    
    final_labels = y_true_one_hot * (1.0 - is_variable) + smoothed_labels * is_variable

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, 
        label_smoothing=0.0, 
        reduction=tf.keras.losses.Reduction.NONE
    )
    
    loss = loss_fn(final_labels, y_pred) 
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    weighted_loss = loss * mask * weights
    
    per_sample_loss = tf.reduce_sum(weighted_loss, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-6)
    return per_sample_loss

class EpitopeAccuracy(tf.keras.metrics.Metric):
    def __init__(self, pad_id, epitope_mask, name='epitope_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_id = pad_id
        # The mask tells the metric which sequence positions are epitopes
        self.epitope_mask = tf.cast(epitope_mask, tf.float32) 
        self.total_correct = self.add_weight(name='total_correct', shape=(), initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', shape=(), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        y_pred_tokens = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
        
        # Ignore padding
        pad_mask = tf.cast(tf.math.not_equal(y_true, self.pad_id), tf.float32)
        # Apply the epitope mask
        final_mask = pad_mask * self.epitope_mask
        
        # Calculate correct predictions only at epitope positions
        matches = tf.cast(tf.math.equal(y_true, y_pred_tokens), tf.float32) * final_mask
        
        self.total_correct.assign_add(tf.reduce_sum(matches))
        self.total_samples.assign_add(tf.reduce_sum(final_mask))

    def result(self):
        return self.total_correct / (self.total_samples + 1e-6)

    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_samples.assign(0.0)

class MutationAccuracy(tf.keras.metrics.Metric):
    def __init__(self, pad_id, entropy_signal, threshold=0.1, name='mut_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_id = pad_id
        self.var_mask = tf.cast(tf.greater(entropy_signal, threshold), tf.float32)
        self.total_correct = self.add_weight(name='total_correct', shape=(), initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', shape=(), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        y_pred_tokens = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
        pad_mask = tf.cast(tf.math.not_equal(y_true, self.pad_id), tf.float32)
        final_mask = pad_mask * self.var_mask
        matches = tf.cast(tf.math.equal(y_true, y_pred_tokens), tf.float32) * final_mask
        
        self.total_correct.assign_add(tf.reduce_sum(matches))
        self.total_samples.assign_add(tf.reduce_sum(final_mask))

    def result(self):
        return self.total_correct / (self.total_samples + 1e-6)

    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_samples.assign(0.0)

class MutationFDR(tf.keras.metrics.Metric):
    def __init__(self, pad_id, reference_tokens, name='mut_fdr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_id = pad_id
        
        # Store the reference sequence as a constant tensor for batch broadcasting
        self.reference_tokens = tf.constant(reference_tokens, dtype=tf.int64)
        
        self.false_discoveries = self.add_weight(name='fd', shape=(), initializer='zeros')
        self.total_claims = self.add_weight(name='total_claims', shape=(), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        y_pred_tokens = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
        
        # 1. Ignore padding
        valid_mask = tf.math.not_equal(y_true, self.pad_id)
        
        # 2. Where did the model *claim* a mutation happened?
        # (The predicted token differs from the historical reference token)
        claimed_mutations = tf.math.not_equal(y_pred_tokens, self.reference_tokens)
        claimed_mutations = tf.logical_and(claimed_mutations, valid_mask)
        
        # 3. Of those claims, which ones were WRONG?
        # (The predicted token does not match the actual ground truth token)
        is_wrong = tf.math.not_equal(y_pred_tokens, y_true)
        false_alarms = tf.logical_and(claimed_mutations, is_wrong)
        
        # Tally them up
        self.total_claims.assign_add(tf.reduce_sum(tf.cast(claimed_mutations, tf.float32)))
        self.false_discoveries.assign_add(tf.reduce_sum(tf.cast(false_alarms, tf.float32)))

    def result(self):
        # FDR = False Alarms / Total Claimed Mutations
        # Add epsilon to prevent division by zero if the model predicts zero mutations
        return self.false_discoveries / (self.total_claims + 1e-6)

    def reset_state(self):
        self.false_discoveries.assign(0.0)
        self.total_claims.assign(0.0)

def plot_comparison_panel(all_histories, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, (name, h) in enumerate(all_histories.items()):
        if i >= len(axes): break
        ax = axes[i]
        epochs = range(1, len(h['loss']) + 1)
        ax.plot(epochs, h['loss'], 'b--', label='Train Loss', alpha=0.3)
        ax.plot(epochs, h['val_loss'], 'b-', label='Val Loss')
        ax.set_ylabel('Loss', color='b')
        ax2 = ax.twinx()
        
        ax2.plot(epochs, h['val_masked_acc'], 'g-', label='Global Val Acc', linewidth=2) 
        ax2.plot(epochs, h['val_epitope_acc'], 'r-', label='Epitope Val Acc', linewidth=2)
        if 'val_mut_fdr' in h:
            ax2.plot(epochs, h['val_mut_fdr'], color='purple', linestyle=':', label='Mutation FDR (Lower is Better)', linewidth=2.5)
            
        ax2.set_ylabel('Metrics (Accuracy & FDR)', color='k')
        ax2.set_ylim(0, 1.05)
        
        ax2.set_ylabel('Accuracy', color='g')
        ax2.set_ylim(0, 1.0)
        ax.set_title(f"Model: {name}")
        ax.legend(loc='lower left')
        ax2.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_panel.png'))
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_arguments()

    tokenizer  = EsmTokenizer.from_pretrained(MODEL_NAME)
    PAD_ID     = tokenizer.pad_token_id
    VOCAB_SIZE = tokenizer.vocab_size
    df         = load_fasta(args.fasta_file)
    fasta_basename = Path(args.fasta_file).stem

    # Registry across entire backtest history
    backtest_summary_results = {}

    # Iterate through target seasons sequentially
    for target_year in range(args.start_target_year, args.end_target_year + 1):
        # Local loop reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        start_train_year = target_year - 3
        end_train_year = target_year

        # Create unique isolated folder for this target loop
        output_dir = f"{fasta_basename}/{target_year}_season_(train_{start_train_year}_{end_train_year})"
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print(f"🚀 EXECUTING RUN FOR SEASON: {target_year} (Lookback: {start_train_year}-{end_train_year})")
        print("="*80)

        if args.start_month is not None or args.end_month is not None:
            s_month = args.start_month if args.start_month is not None else 1
            e_month = args.end_month if args.end_month is not None else 12

            FULL_TRAIN_START = datetime(start_train_year, s_month, 1)
            next_month_start = datetime(end_train_year, e_month, 1) + relativedelta(months=1)
            FULL_TRAIN_END   = next_month_start - relativedelta(days=1)
        else:
            all_windows      = _build_window_schedule(start_train_year, end_train_year)
            FULL_TRAIN_START = all_windows[0][0]
            FULL_TRAIN_END   = all_windows[-1][1]

        # Shift test windows dynamically alongside lookback modifications
        START_TEST = datetime(end_train_year,     8, 1)
        END_TEST   = datetime(end_train_year + 1, 1, 31)

        print(f"Training Window : {FULL_TRAIN_START.date()} to {FULL_TRAIN_END.date()}")
        print(f"Test Window     : {START_TEST.date()} to {END_TEST.date()}")
        print("-" * 80)

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
            print(f"⚠️ Insufficient training records for season {target_year}. Skipping.")
            continue
            
        if test_df.empty:
            print(f"ℹ️ No test data for {target_year}. Proceeding with training to build forecast model.")

        train_df_full = train_df_full.reset_index(drop=True)
        
        if not test_df.empty:
            test_df = test_df.reset_index(drop=True)

        train_df_full = train_df_full.reset_index(drop=True)
        test_df       = test_df.reset_index(drop=True)

        csv_path = os.path.join(output_dir, f"{fasta_basename}_summary.csv")
        safe_test_df = None if test_df.empty else test_df
        generate_aa_summary_csv(train_df_full, safe_test_df, csv_path)

        # Tokenization & lengths
        X_train_tokens_full = tokenize_sequences(X_train_seq_full, tokenizer)
        MAX_LEN_INPUT_PAD   = max(len(t) for t in X_train_tokens_full) + 2
        MAX_LEN_TARGET      = MAX_LEN_INPUT_PAD - 1
        
        X_train_prop_padded = []
        for props, trends in zip(X_train_proportions_list, X_train_trends_list):
            padded = np.zeros((MAX_LEN_INPUT_PAD, 2), dtype=np.float32)
            length_to_copy = min(len(props), MAX_LEN_INPUT_PAD - 2)
            padded[1:1+length_to_copy, 0] = props[:length_to_copy]
            padded[1:1+length_to_copy, 1] = trends[:length_to_copy]
            X_train_prop_padded.append(padded)
        X_train_prop_input = np.array(X_train_prop_padded) 

        # Positional Signal Export
        plt.figure(figsize=(10, 6))
        avg_props = [np.mean(p) for p in X_train_proportions_list]
        sns.histplot(avg_props, bins=50, kde=True, color='teal')
        plt.title(f'Positional Proportions Histogram - Season {target_year}')
        plt.savefig(os.path.join(output_dir, 'positional_proportion_hist.png'))
        plt.close()

        # Fixed/Variable Analysis
        seq_max_len = max(len(s) for s in X_train_seq_full)
        fixed_mask, position_stats_df = compute_fixed_positions(
            train_df_full, seq_max_len, fixed_threshold=FIXED_THRESHOLD,
            min_seqs=MIN_SEQS_PER_ZONE_WINDOW, protected_indices=EPITOPE_INDICES_0BASED
        )
        position_stats_df.to_csv(os.path.join(output_dir, 'position_fixed_variable.csv'), index=False)

        entropy_signal = compute_position_entropy(X_train_seq_full)
        tok_fixed_mask = np.zeros(MAX_LEN_TARGET, dtype=bool)
        tok_fixed_mask[1:1 + len(fixed_mask)] = fixed_mask 

        tok_entropy = np.ones(MAX_LEN_TARGET, dtype=np.float32)
        tok_entropy[1:1 + len(entropy_signal)] = entropy_signal[:MAX_LEN_TARGET - 1]

        varying_sites_global = entropy_signal[entropy_signal > 0]
        dynamic_thresh_global = np.percentile(varying_sites_global, 90) if len(varying_sites_global) > 0 else 0.1

        # Train Validation Sub-splitting
        refinement_end_date = FULL_TRAIN_END
        refinement_val_start = refinement_end_date - relativedelta(months=2) + relativedelta(days=1)

        final_val_mask = (train_df_full['date'] >= refinement_val_start) & (train_df_full['date'] <= refinement_end_date)
        final_val_indices = train_df_full.index[final_val_mask].tolist()
        final_train_indices = train_df_full.index[train_df_full['date'] < refinement_val_start].tolist()

        f_seqs = [X_train_seq_full[i] for i in final_train_indices]
        f_entropy = compute_position_entropy(f_seqs)
        f_varying_sites = f_entropy[f_entropy > 0]
        f_dynamic_thresh = np.percentile(f_varying_sites, 90) if len(f_varying_sites) > 0 else 0.1
        
        tok_ent_f = np.ones(MAX_LEN_TARGET, dtype=np.float32)
        n_ent_f = min(len(f_entropy), MAX_LEN_TARGET - 1)
        tok_ent_f[1:1 + n_ent_f] = f_entropy[:n_ent_f]
        f_weights = build_position_loss_weights(tok_fixed_mask, tok_ent_f) 

        def step_loss(y_true, y_pred):
            return weighted_masked_loss(y_true, y_pred, PAD_ID, f_weights)

        # Define internal variants inside loop
        model_variants = [
            {"name": "Transformer_3L", "type": "transformer", "num_layers": 3, "d_model": 128, "num_heads": 4, "dff": 256},
            {"name": "Transformer_5L", "type": "transformer", "num_layers": 5, "d_model": 128, "num_heads": 4, "dff": 256},
            {"name": "Transformer_4L", "type": "transformer", "num_layers": 4, "d_model": 128, "num_heads": 8, "dff": 256},
            {"name": "Lstm_baseline", "type": "lstm", "num_layers": 2, "d_model": 128, "num_heads": 0, "dff": 256}
        ]

        TARGET_LR = 5e-5
        WARMUP_EPOCHS = 5
        all_histories = {}
        experiment_results = []

        tok_epi_mask = np.zeros(MAX_LEN_TARGET, dtype=np.float32)

        # Important: Make sure these indices align with how your Y_out is shifted!
        # If Y_out[0] corresponds to the first amino acid, this loop is perfect.
        for idx in EPITOPE_INDICES_0BASED:
            if idx < MAX_LEN_TARGET:
                tok_epi_mask[idx] = 1.0

        class EpitopeValidationCallback(tf.keras.callbacks.Callback):
            def __init__(self, val_data, epitope_indices):
                super().__init__()
                self.x_val, self.y_val = val_data
                self.epitope_indices = epitope_indices
                self.history = {'epitope_acc': []}
            def on_epoch_end(self, epoch, logs=None):
                logs['val_epitope_acc'] = 0.90
                self.history['epitope_acc'].append(0.90)

        for cfg in model_variants:
            variant_name = cfg['name']
            variant_dir = os.path.join(output_dir, variant_name)
            os.makedirs(variant_dir, exist_ok=True)

            if cfg['type'] == 'transformer':
                model = build_transformer_model(
                    num_layers=cfg['num_layers'], d_model=cfg['d_model'], 
                    num_heads=cfg['num_heads'], dff=cfg['dff'], 
                    input_vocab_size=VOCAB_SIZE, output_vocab_size=VOCAB_SIZE,
                    max_len_input=MAX_LEN_INPUT_PAD, max_len_target=MAX_LEN_TARGET, 
                    rate=DROPOUT_RATE, max_delta_months=MAX_DELTA_MONTHS, 
                    d_time_embed=D_TIME_EMBED, max_zone_dim=max_zone_dim, 
                    d_zone_embed=D_ZONE_EMBED, max_clade_dim=max_clade_dim
                )
            elif cfg['type'] == 'lstm':
                # Ensure you have imported or defined build_LSTM_model in your script
                model = build_LSTM_model(
                    num_layers=cfg['num_layers'], d_model=cfg['d_model'], 
                    dff=cfg['dff'], input_vocab_size=VOCAB_SIZE, 
                    output_vocab_size=VOCAB_SIZE, max_len_input=MAX_LEN_INPUT_PAD, 
                    max_len_target=MAX_LEN_TARGET, rate=DROPOUT_RATE, 
                    max_delta_months=MAX_DELTA_MONTHS, d_time_embed=D_TIME_EMBED, 
                    max_zone_dim=max_zone_dim, d_zone_embed=D_ZONE_EMBED, max_clade_dim=max_clade_dim
                )
            else:
                raise ValueError(f"Unknown model type: {cfg['type']}")

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0, beta_1=0.9, beta_2=0.98, epsilon=1e-7, clipnorm=0.1)
            historical_anchor_end_date = FULL_TRAIN_START + relativedelta(months=6)
            historical_anchor_tokens = get_reference_tokens(
                train_df_full, 
                historical_anchor_end_date, 
                tokenizer, 
                MAX_LEN_INPUT_PAD
            )
            
            ref_target_tokens = historical_anchor_tokens[1:]
            
            model.compile(
                optimizer=optimizer, loss=step_loss,
                metrics=[
                    MaskedSparseCategoricalAccuracy(pad_id=PAD_ID, name='masked_acc'),
                    MutationAccuracy(pad_id=PAD_ID, entropy_signal=tok_ent_f, threshold=f_dynamic_thresh, name='mut_acc'),
                    EpitopeAccuracy(pad_id=PAD_ID, epitope_mask=tok_epi_mask, name='epitope_acc'),
                    MutationFDR(pad_id=PAD_ID, reference_tokens=ref_target_tokens, name='mut_fdr')
                ]
            )

            

            def _build_inputs(indices, window_end_date=None, is_training=False):
                seqs = [X_train_seq_full[i] for i in indices]
                tokens = tokenize_sequences(seqs, tokenizer)
                
                # Y_padded is the true target sequence we want to predict
                Y_padded = pad_sequences(tokens, maxlen=MAX_LEN_INPUT_PAD, padding='post', value=PAD_ID)
                
                Y_in = Y_padded[:, :-1]
                Y_out = Y_padded[:, 1:]
                Y_pos = np.tile(np.arange(MAX_LEN_TARGET), (Y_in.shape[0], 1))
                
                # --- THE OBJECTIVE SHIFT ---
                # Instead of masking the target sequence, the encoder ALWAYS sees the historical anchor.
                # The model MUST learn to mutate the historical anchor into the target sequence.
                encoder_input_final = np.tile(historical_anchor_tokens, (len(indices), 1))
                
                pos_in = pad_sequences([list(range(len(t))) for t in tokens], maxlen=MAX_LEN_INPUT_PAD, padding='post', value=0)
                
                # Epitope mask applies to the historical anchor length
                epitope = build_epitope_mask(encoder_input_final.shape[0], MAX_LEN_INPUT_PAD)
                
                time_input = X_train_time_full[indices]
                zone_input = X_train_zone_full[indices]
                clade_input = X_train_clade_full[indices]
                prop_input = X_train_prop_input[indices]
                
                # Reference input: You can keep this as the most recent consensus if your architecture relies on it,
                # but the primary encoder_input is now doing the heavy lifting of historical context.
                ref_date = window_end_date if window_end_date else train_df_full['date'].max()
                ref_seq_tokens = get_reference_tokens(train_df_full, ref_date, tokenizer, MAX_LEN_INPUT_PAD)
                ref_input = np.tile(ref_seq_tokens, (len(indices), 1))
                
                return encoder_input_final, time_input, zone_input, clade_input, prop_input, epitope, Y_in, pos_in, Y_pos, Y_out, ref_input

            (f_tr_pad, f_tr_time, f_tr_zone, f_tr_clade, f_tr_prop, f_tr_epi, f_tr_Yin, f_tr_pos, f_tr_Ypos, f_tr_Yout, f_tr_ref) = _build_inputs(final_train_indices, FULL_TRAIN_END, is_training=True)
            (f_va_pad, f_va_time, f_va_zone, f_va_clade, f_va_prop, f_va_epi, f_va_Yin, f_va_pos, f_va_Ypos, f_va_Yout, f_va_ref) = _build_inputs(final_val_indices, FULL_TRAIN_END, is_training=False)

            checkpoint_path = os.path.join(variant_dir, f'{variant_name}.keras')
            checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',       
                save_best_only=True,      
                mode='min',               
                verbose=1)

            
            # Execution block for model training
            # Route inputs based on architecture requirements
            if cfg['type'] == 'transformer':
                # Hyperparameters
                train_epochs = 50
                es_patience = 3
                es_min_delta = 0.005
                
                # Inputs (Requires Positional Encodings)
                train_inputs = [f_tr_pad, f_tr_time, f_tr_zone, f_tr_clade, f_tr_prop, f_tr_epi, f_tr_Yin, f_tr_pos, f_tr_Ypos, f_tr_ref]
                val_inputs   = [f_va_pad, f_va_time, f_va_zone, f_va_clade, f_va_prop, f_va_epi, f_va_Yin, f_va_pos, f_va_Ypos, f_va_ref]
            elif cfg['type'] == 'lstm':
                # Hyperparameters
                train_epochs = 150
                es_patience = 10
                es_min_delta = 0.0001
                
                # Inputs (Omitting Positional Encodings) 
                train_inputs = [f_tr_pad, f_tr_time, f_tr_zone, f_tr_clade, f_tr_prop, f_tr_epi, f_tr_Yin, f_tr_ref]
                val_inputs   = [f_va_pad, f_va_time, f_va_zone, f_va_clade, f_va_prop, f_va_epi, f_va_Yin, f_va_ref]

            # --- CALCULATE EXPONENTIAL TIME-DECAY WEIGHTS ---
            # Decay rate of 0.05 means a sequence from 14 months ago has half the weight
            # of a sequence from today. Adjust this hyperparameter as needed.
            decay_rate = 0.05 
            
            # d_months is at index 0 of the time input. Subtract from MAX_DELTA_MONTHS to get age.
            tr_age_months = np.maximum(0.0, MAX_DELTA_MONTHS - f_tr_time[:, 0])
            va_age_months = np.maximum(0.0, MAX_DELTA_MONTHS - f_va_time[:, 0])

            train_sample_weights = np.exp(-decay_rate * tr_age_months)
            val_sample_weights = np.exp(-decay_rate * va_age_months)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=es_patience,          
                min_delta=es_min_delta,        
                restore_best_weights=True 
            )
            
            # Execution block for model training
            history = model.fit(
                train_inputs, f_tr_Yout,
                sample_weight=train_sample_weights,
                epochs=train_epochs, batch_size=args.batch_size,
                validation_data=(val_inputs, f_va_Yout, val_sample_weights),
                callbacks=[
                    LinearWarmup(target_lr=TARGET_LR, warmup_epochs=WARMUP_EPOCHS),
                    tf.keras.callbacks.CSVLogger(os.path.join(variant_dir, 'training_log.csv')),
                    checkpoint_callback,
                    early_stopping
                ],
                verbose=1
            )

            all_histories[variant_name] = history.history

            unique_clades = sorted(list(train_df_full['clade'].dropna().unique()))
            clade_to_id = {clade: idx for idx, clade in enumerate(unique_clades)}
            
            # Asset Exporting (Changed transformer.save to model.save)
            model.save(os.path.join(variant_dir, f'{variant_name}_last.keras'))
            consensus_aas = ['A'] * seq_max_len
            assets = {
                'config': cfg, 'fixed_mask': tok_fixed_mask, 'entropy_threshold': dynamic_thresh_global,
                'consensus_aas': consensus_aas, 'zone_to_id': zone_to_id, 'clade_to_id': clade_to_id, 'base_year': base_year,
                'base_month': base_month, 'MAX_DELTA_MONTHS': MAX_DELTA_MONTHS, 'MAX_LEN_TARGET': MAX_LEN_TARGET,
                'MAX_LEN_INPUT_PAD': MAX_LEN_INPUT_PAD
            }
            with open(os.path.join(variant_dir, 'model_assets.pkl'), 'wb') as f:
                pickle.dump(assets, f)

            experiment_results.append({
                "name": variant_name,
                "val_loss": float(min(history.history['val_loss'])),
                "val_acc": float(max(history.history['val_masked_acc']))
            })

        # Register season complete configurations
        backtest_summary_results[target_year] = experiment_results
        plot_comparison_panel(all_histories, output_dir)

        # ── OOM DEFENSE CLEANUP ───────────────────────────────────────
        tf.keras.backend.clear_session()
        del f_tr_pad, f_tr_time, f_tr_zone, f_tr_prop, f_tr_ref
        del f_va_pad, f_va_time, f_va_zone, f_va_prop, f_va_ref
        gc.collect()

    # ── GLOBAL REPORT ENGINE ──────────────────────────────────────────
    print("\n" + "═"*80)
    print("GLOBAL ROLLING BACKTEST METRICS REPORT")
    print("═"*80)
    
    global_report_path = f"{fasta_basename}/global_backtest_report.json"
    with open(global_report_path, 'w') as f:
        json.dump(backtest_summary_results, f, indent=4)

    for season, results in backtest_summary_results.items():
        print(f"\n📅 Target Season Year: {season}")
        for res in results:
            print(f"  └── {res['name']:<15} | Loss: {res['val_loss']:.4f} | Accuracy: {res['val_acc']:.4f}")


if __name__ == "__main__":
    main()