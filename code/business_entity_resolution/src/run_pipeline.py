#!/usr/bin/env python3
"""
Main pipeline for Business Entity Resolution.

Usage:
    python run_pipeline.py --mode train        # Train on training data, evaluate
    python run_pipeline.py --mode predict      # Predict on test data
    python run_pipeline.py --mode all          # Train + predict
    python run_pipeline.py --mode eda          # Run EDA only
    python run_pipeline.py --sample 1000       # Quick smoke test with N S1 entities
"""

import argparse
import csv
import gc
import logging
import os
import pickle
import sys
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, "../../.."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_record, normalize_country
from blocking import BlockingEngine, sparse_cosine_topk
from features import compute_pair_features, compute_features_batch, FEATURE_NAMES
from metric import macro_f05, macro_f05_detailed, f05_per_entity
from decide import make_predictions, tune_thresholds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT, 'artifacts', 'pipeline.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Paths
TRAIN_DIR = os.path.join(ROOT, "dataset/train")
TEST_DIR = os.path.join(ROOT, "dataset/test")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
OUTPUT_DIR = os.path.join(ROOT, "output")
REPORTS_DIR = os.path.join(ROOT, "reports")

for d in [ARTIFACTS_DIR, OUTPUT_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
np.random.seed(SEED)


def load_tsv(path):
    """Load a TSV file."""
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_values=[])


def load_ground_truth(path):
    """Load ground truth as dict {s1_id: set of matched ids}."""
    df = load_tsv(path)
    gt = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids'].strip()
        gt[s1_id] = set(matched.split(',')) if matched else set()
    return gt


def normalize_dataframe(df, desc=""):
    """Normalize all records in a DataFrame. Returns list of record dicts."""
    logger.info(f"  Normalizing {len(df):,} records ({desc})...")
    t0 = time.time()
    records = []
    for _, row in df.iterrows():
        rec = normalize_record(
            row['entity_id'], 
            row['business_name'], 
            row['business_address'],
            row['country']
        )
        records.append(rec)
    logger.info(f"  Done in {time.time()-t0:.1f}s")
    return records


def normalize_dataframe_fast(df, desc=""):
    """Faster normalization using vectorized operations where possible."""
    logger.info(f"  Fast-normalizing {len(df):,} records ({desc})...")
    t0 = time.time()
    
    # Pre-allocate
    records = []
    total = len(df)
    
    entity_ids = df['entity_id'].values
    names = df['business_name'].values
    addresses = df['business_address'].values
    countries = df['country'].values
    
    for i in range(total):
        if i > 0 and i % 500000 == 0:
            logger.info(f"    {i:,}/{total:,} ({100*i/total:.0f}%)")
        rec = normalize_record(entity_ids[i], names[i], addresses[i], countries[i])
        records.append(rec)
    
    logger.info(f"  Done in {time.time()-t0:.1f}s ({total/max(time.time()-t0, 0.01):.0f} rec/s)")
    return records


def run_blocking_by_country(s1_records, s23_records, blocker, sample_n=None):
    """Run blocking within each country, then combine results.
    
    Args:
        s1_records: list of normalized S1 record dicts
        s23_records: list of normalized S2/S3 record dicts
        blocker: BlockingEngine instance
        sample_n: if set, subsample S1 to this many per country
    
    Returns:
        dict {s1_global_idx: [(s23_global_idx, score), ...]}
    """
    # Group by country
    s1_by_country = defaultdict(list)  # country -> [(global_idx, record)]
    s23_by_country = defaultdict(list)
    
    for i, rec in enumerate(s1_records):
        c = rec.get('country', '') or 'UNKNOWN'
        s1_by_country[c].append((i, rec))
    
    for i, rec in enumerate(s23_records):
        c = rec.get('country', '') or 'UNKNOWN'
        s23_by_country[c].append((i, rec))
    
    all_countries = set(s1_by_country.keys()) | set(s23_by_country.keys())
    logger.info(f"Countries found: {sorted(all_countries)}")
    
    global_candidates = {}
    
    for country in sorted(all_countries):
        s1_items = s1_by_country.get(country, [])
        s23_items = s23_by_country.get(country, [])
        
        if not s1_items or not s23_items:
            logger.info(f"  Skipping {country}: {len(s1_items)} S1, {len(s23_items)} S2/S3")
            continue
        
        # Subsample if needed
        if sample_n and len(s1_items) > sample_n:
            np.random.seed(SEED)
            indices = np.random.choice(len(s1_items), sample_n, replace=False)
            s1_items = [s1_items[i] for i in indices]
        
        # Extract records for blocking
        s1_global_idxs = [item[0] for item in s1_items]
        s1_recs = [item[1] for item in s1_items]
        s23_global_idxs = [item[0] for item in s23_items]
        s23_recs = [item[1] for item in s23_items]
        
        # Run blocking
        local_candidates = blocker.generate_candidates(s1_recs, s23_recs, country=country)
        
        # Map back to global indices
        for local_s1_idx, cand_list in local_candidates.items():
            global_s1_idx = s1_global_idxs[local_s1_idx]
            global_cands = [(s23_global_idxs[local_s23_idx], score) 
                           for local_s23_idx, score in cand_list]
            global_candidates[global_s1_idx] = global_cands
        
        # Free memory
        del local_candidates, s1_recs, s23_recs
        gc.collect()
    
    return global_candidates


def compute_all_features(s1_records, s23_records, candidates):
    """Compute features for all candidate pairs.
    
    Returns:
        (s1_indices, s23_indices, feature_matrix, feature_names)
    """
    logger.info(f"  Computing features for {sum(len(v) for v in candidates.values()):,} pairs...")
    t0 = time.time()
    
    s1_indices = []
    s23_indices = []
    all_features = []
    
    total_s1 = len(candidates)
    processed = 0
    
    for s1_idx, cand_list in candidates.items():
        if not cand_list:
            continue
        
        s1_rec = s1_records[s1_idx]
        n_cands = len(cand_list)
        scores = [score for _, score in cand_list]
        top1_score = scores[0] if scores else 0
        top2_score = scores[1] if len(scores) > 1 else 0
        
        for rank, (s23_idx, block_score) in enumerate(cand_list):
            s23_rec = s23_records[s23_idx]
            feats = compute_pair_features(
                s1_rec, s23_rec,
                blocking_score=block_score,
                candidate_rank=rank,
                n_candidates=n_cands,
                top1_score=top1_score,
                top2_score=top2_score,
            )
            s1_indices.append(s1_idx)
            s23_indices.append(s23_idx)
            all_features.append(feats)
        
        processed += 1
        if processed % 100000 == 0:
            logger.info(f"    Features: {processed:,}/{total_s1:,} S1 entities")
    
    # Convert to array
    if not all_features:
        return np.array([]), np.array([]), np.empty((0, 0)), []
    
    feature_names = sorted(all_features[0].keys())
    X = np.array([[d.get(f, 0.0) for f in feature_names] for d in all_features], dtype=np.float32)
    
    logger.info(f"  Features done: {X.shape[0]:,} pairs x {X.shape[1]} features in {time.time()-t0:.1f}s")
    return np.array(s1_indices), np.array(s23_indices), X, feature_names


def train_model(X_train, y_train, feature_names, n_folds=3, groups=None):
    """Train LightGBM model with GroupKFold cross-validation.
    
    Returns:
        (model, oof_predictions, fold_models)
    """
    logger.info(f"  Training LightGBM: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
        'scale_pos_weight': sum(y_train == 0) / max(sum(y_train == 1), 1),
    }
    
    oof_preds = np.zeros(len(y_train))
    fold_models = []
    
    if groups is not None:
        kf = GroupKFold(n_splits=n_folds)
        splits = list(kf.split(X_train, y_train, groups))
    else:
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        splits = list(kf.split(X_train, y_train))
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  Fold {fold_idx+1}/{n_folds}: train={len(train_idx):,}, val={len(val_idx):,}")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, free_raw_data=False)
        
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50),
            ],
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        fold_models.append(model)
        
        logger.info(f"  Fold {fold_idx+1} best iteration: {model.best_iteration}")
    
    return fold_models, oof_preds


def evaluate_on_validation(oof_preds, s1_indices, s23_indices, s1_records, s23_records,
                           ground_truth, all_s1_ids, s1_countries=None):
    """Evaluate OOF predictions with threshold tuning.
    
    Returns:
        best_params dict including 'score'
    """
    # Create scored pairs
    scored_pairs = []
    for i in range(len(oof_preds)):
        s1_id = s1_records[s1_indices[i]]['entity_id']
        s23_id = s23_records[s23_indices[i]]['entity_id']
        scored_pairs.append((s1_id, s23_id, oof_preds[i]))
    
    # Tune thresholds
    logger.info("  Tuning thresholds...")
    best = tune_thresholds(
        scored_pairs, ground_truth, all_s1_ids, macro_f05,
        threshold_grid=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        margin_grid=[0.0, 0.02, 0.05, 0.1, 0.15],
    )
    
    logger.info(f"  Best params: T_first={best['threshold_first']}, "
                f"T_extra={best['threshold_extra']}, margin={best['competition_margin']}")
    
    # Get detailed metrics with best params
    preds = make_predictions(
        scored_pairs, all_s1_ids,
        threshold_first=best['threshold_first'],
        threshold_extra=best['threshold_extra'],
        competition_margin=best['competition_margin'],
    )
    
    detailed = macro_f05_detailed(preds, ground_truth, s1_countries)
    best.update(detailed)
    
    logger.info(f"  Validation macro-F0.5: {detailed['overall']:.4f}")
    logger.info(f"    Singleton F0.5: {detailed['singleton']:.4f}")
    logger.info(f"    Non-singleton F0.5: {detailed['non_singleton']:.4f}")
    if 'per_country' in detailed:
        for c, s in detailed['per_country'].items():
            logger.info(f"    {c} F0.5: {s:.4f}")
    
    return best


def write_output(predictions, candidates_dict, output_dir, s1_ids_ordered):
    """Write matching_results.tsv and candidate_pairs.tsv.
    
    Args:
        predictions: dict {s1_id: set of matched s23_ids}
        candidates_dict: dict {s1_id: set of candidate s23_ids}
        output_dir: output directory
        s1_ids_ordered: list of all S1 IDs in order
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Write matching_results.tsv
    match_path = os.path.join(output_dir, "matching_results.tsv")
    with open(match_path, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids_ordered:
            matched = predictions.get(s1_id, set())
            matched_str = ','.join(sorted(matched)) if matched else ''
            f.write(f"{s1_id}\t{matched_str}\n")
    
    logger.info(f"  Written {match_path}: {len(s1_ids_ordered):,} rows")
    
    # Write candidate_pairs.tsv
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(cand_path, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids_ordered:
            cands = candidates_dict.get(s1_id, set())
            cand_str = ','.join(sorted(cands)) if cands else ''
            f.write(f"{s1_id}\t{cand_str}\n")
    
    logger.info(f"  Written {cand_path}: {len(s1_ids_ordered):,} rows")


def run_train(args):
    """Training pipeline: normalize, block, feature, train, evaluate."""
    t_start = time.time()
    
    # Load data
    logger.info("=" * 60)
    logger.info("LOADING TRAINING DATA")
    logger.info("=" * 60)
    
    s1_df = load_tsv(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    s2_df = load_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"))
    s3_df = load_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    
    # Combine S2 and S3
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    logger.info(f"  S1: {len(s1_df):,}, S2+S3: {len(s23_df):,}, GT: {len(gt):,}")
    
    # Sample if requested
    if args.sample:
        logger.info(f"  SAMPLING: {args.sample} S1 entities")
        np.random.seed(SEED)
        sample_ids = np.random.choice(s1_df['entity_id'].values, 
                                       min(args.sample, len(s1_df)), replace=False)
        sample_ids_set = set(sample_ids)
        s1_df = s1_df[s1_df['entity_id'].isin(sample_ids_set)].reset_index(drop=True)
        
        # Get all matched S2/S3 IDs for sampled S1
        relevant_s23 = set()
        for s1_id in sample_ids_set:
            relevant_s23.update(gt.get(s1_id, set()))
        
        # Keep S23 records that are matched + a sample of unmatched for hard negatives
        s23_matched = s23_df[s23_df['entity_id'].isin(relevant_s23)]
        
        # For blocking, we need the pool of same-country records
        s1_countries_set = set(s1_df['country'].values)
        s23_same_country = s23_df[s23_df['country'].isin(s1_countries_set)]
        
        # Sample unmatched
        s23_unmatched = s23_same_country[~s23_same_country['entity_id'].isin(relevant_s23)]
        if len(s23_unmatched) > args.sample * 20:
            s23_unmatched = s23_unmatched.sample(n=args.sample * 20, random_state=SEED)
        
        s23_df = pd.concat([s23_matched, s23_unmatched], ignore_index=True).drop_duplicates(subset=['entity_id'])
        gt = {k: v for k, v in gt.items() if k in sample_ids_set}
        
        logger.info(f"  After sampling: S1={len(s1_df):,}, S2+S3={len(s23_df):,}")
    
    # Normalize
    logger.info("=" * 60)
    logger.info("NORMALIZING RECORDS")
    logger.info("=" * 60)
    
    cache_path = os.path.join(ARTIFACTS_DIR, f"normalized_train{'_sample' + str(args.sample) if args.sample else ''}.pkl")
    
    if os.path.exists(cache_path) and not args.no_cache:
        logger.info(f"  Loading cached normalized records from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        s1_records = cached['s1_records']
        s23_records = cached['s23_records']
    else:
        s1_records = normalize_dataframe_fast(s1_df, "S1 train")
        s23_records = normalize_dataframe_fast(s23_df, "S2+S3 train")
        
        with open(cache_path, 'wb') as f:
            pickle.dump({'s1_records': s1_records, 's23_records': s23_records}, f)
        logger.info(f"  Cached to {cache_path}")
    
    # Build indices
    s1_ids = [r['entity_id'] for r in s1_records]
    s23_ids = [r['entity_id'] for r in s23_records]
    s1_id_set = set(s1_ids)
    s1_countries = {r['entity_id']: r['country'] for r in s1_records}
    
    # Blocking
    logger.info("=" * 60)
    logger.info("BLOCKING / CANDIDATE GENERATION")
    logger.info("=" * 60)
    
    blocker = BlockingEngine(top_k_per_channel=50, top_k_final=100)
    candidates = run_blocking_by_country(s1_records, s23_records, blocker)
    
    # Evaluate blocking recall
    blocking_eval = blocker.evaluate_blocking_recall(
        candidates, s1_records, s23_records, gt, s1_ids, s23_ids
    )
    logger.info(f"  Blocking recall: {blocking_eval['recall']:.4f} "
                f"({blocking_eval['found_pairs']:,}/{blocking_eval['total_true_pairs']:,})")
    if blocking_eval['missed_examples']:
        logger.info(f"  Missed examples: {blocking_eval['missed_examples'][:5]}")
    
    # Feature computation
    logger.info("=" * 60)
    logger.info("FEATURE COMPUTATION")
    logger.info("=" * 60)
    
    s1_idx_arr, s23_idx_arr, X, feature_names = compute_all_features(
        s1_records, s23_records, candidates
    )
    
    # Create labels
    s23_id_lookup = {eid: i for i, eid in enumerate(s23_ids)}
    y = np.zeros(len(s1_idx_arr), dtype=np.float32)
    for i in range(len(s1_idx_arr)):
        s1_id = s1_ids[s1_idx_arr[i]]
        s23_id = s23_ids[s23_idx_arr[i]]
        if s23_id in gt.get(s1_id, set()):
            y[i] = 1.0
    
    logger.info(f"  Labels: {int(y.sum()):,} positives / {len(y):,} total "
                f"({y.mean()*100:.2f}%)")
    
    # Train model with GroupKFold on S1 entities
    logger.info("=" * 60)
    logger.info("MODEL TRAINING")
    logger.info("=" * 60)
    
    groups = s1_idx_arr  # Group by S1 entity
    fold_models, oof_preds = train_model(X, y, feature_names, n_folds=3, groups=groups)
    
    # Evaluate
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    eval_results = evaluate_on_validation(
        oof_preds, s1_idx_arr, s23_idx_arr, s1_records, s23_records,
        gt, s1_id_set, s1_countries
    )
    
    # Train final model on all data
    logger.info("=" * 60)
    logger.info("TRAINING FINAL MODEL")
    logger.info("=" * 60)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 100,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
        'scale_pos_weight': sum(y == 0) / max(sum(y == 1), 1),
    }
    
    # Use average best iteration from folds
    avg_best_iter = int(np.mean([m.best_iteration for m in fold_models]))
    
    train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
    final_model = lgb.train(params, train_data, num_boost_round=avg_best_iter)
    
    # Save everything
    model_path = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_names': feature_names,
            'eval_results': eval_results,
            'blocking_eval': blocking_eval,
        }, f)
    logger.info(f"  Model saved to {model_path}")
    
    # Feature importance
    importance = final_model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    logger.info("  Top 15 features by gain:")
    for name, imp in feat_imp[:15]:
        logger.info(f"    {name}: {imp:.1f}")
    
    elapsed = time.time() - t_start
    logger.info(f"\n  Training complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"  Validation macro-F0.5: {eval_results['overall']:.4f}")
    
    return eval_results


def run_predict(args):
    """Prediction pipeline: load model, normalize test data, block, feature, predict, write output."""
    t_start = time.time()
    
    # Load model
    logger.info("=" * 60)
    logger.info("LOADING MODEL")
    logger.info("=" * 60)
    
    model_path = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    
    model = saved['model']
    feature_names = saved['feature_names']
    eval_results = saved['eval_results']
    
    logger.info(f"  Model loaded. Validation F0.5: {eval_results['overall']:.4f}")
    logger.info(f"  Thresholds: T_first={eval_results['threshold_first']}, "
                f"T_extra={eval_results['threshold_extra']}, "
                f"margin={eval_results['competition_margin']}")
    
    # Load test data
    logger.info("=" * 60)
    logger.info("LOADING TEST DATA")
    logger.info("=" * 60)
    
    s1_df = load_tsv(os.path.join(TEST_DIR, "test_source1.tsv"))
    s2_df = load_tsv(os.path.join(TEST_DIR, "test_source2.tsv"))
    s3_df = load_tsv(os.path.join(TEST_DIR, "test_source3.tsv"))
    
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    logger.info(f"  Test: S1={len(s1_df):,}, S2+S3={len(s23_df):,}")
    
    # Sample if requested
    if args.sample:
        logger.info(f"  SAMPLING: {args.sample} S1 entities for test")
        np.random.seed(SEED + 1)
        sample_ids = np.random.choice(s1_df['entity_id'].values,
                                       min(args.sample, len(s1_df)), replace=False)
        s1_df = s1_df[s1_df['entity_id'].isin(set(sample_ids))].reset_index(drop=True)
        # Keep all S23 for same countries
        s1_countries_set = set(s1_df['country'].values)
        s23_df = s23_df[s23_df['country'].isin(s1_countries_set)].reset_index(drop=True)
    
    # Normalize
    logger.info("=" * 60)
    logger.info("NORMALIZING TEST RECORDS")
    logger.info("=" * 60)
    
    cache_path = os.path.join(ARTIFACTS_DIR, f"normalized_test{'_sample' + str(args.sample) if args.sample else ''}.pkl")
    
    if os.path.exists(cache_path) and not args.no_cache:
        logger.info(f"  Loading cached test records from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        s1_records = cached['s1_records']
        s23_records = cached['s23_records']
    else:
        s1_records = normalize_dataframe_fast(s1_df, "S1 test")
        s23_records = normalize_dataframe_fast(s23_df, "S2+S3 test")
        
        with open(cache_path, 'wb') as f:
            pickle.dump({'s1_records': s1_records, 's23_records': s23_records}, f)
    
    s1_ids = [r['entity_id'] for r in s1_records]
    s23_ids = [r['entity_id'] for r in s23_records]
    
    # For full test: use all S1 from original file (not just sampled)
    all_s1_df = load_tsv(os.path.join(TEST_DIR, "test_source1.tsv"))
    all_s1_ids = all_s1_df['entity_id'].tolist()
    all_s1_id_set = set(all_s1_ids)
    
    # Blocking
    logger.info("=" * 60)
    logger.info("BLOCKING TEST DATA")
    logger.info("=" * 60)
    
    blocker = BlockingEngine(top_k_per_channel=50, top_k_final=100)
    candidates = run_blocking_by_country(s1_records, s23_records, blocker)
    
    total_cand_pairs = sum(len(v) for v in candidates.values())
    logger.info(f"  Total candidate pairs: {total_cand_pairs:,}")
    
    # Feature computation & prediction - process by country chunks to manage memory
    logger.info("=" * 60)
    logger.info("FEATURE COMPUTATION & PREDICTION")
    logger.info("=" * 60)
    
    all_scored_pairs = []
    candidates_dict = defaultdict(set)  # For candidate_pairs.tsv
    
    # Process in chunks of S1 entities
    chunk_size = 50000
    s1_idx_list = sorted(candidates.keys())
    
    for chunk_start in range(0, len(s1_idx_list), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(s1_idx_list))
        chunk_s1_idxs = s1_idx_list[chunk_start:chunk_end]
        
        logger.info(f"  Processing chunk {chunk_start//chunk_size + 1}: "
                    f"S1 indices {chunk_start}-{chunk_end}")
        
        chunk_candidates = {idx: candidates[idx] for idx in chunk_s1_idxs}
        
        s1_idx_arr, s23_idx_arr, X, _ = compute_all_features(
            s1_records, s23_records, chunk_candidates
        )
        
        if len(X) == 0:
            continue
        
        # Predict
        # Ensure features are in same order as training
        preds = model.predict(X)
        
        for i in range(len(preds)):
            s1_id = s1_records[s1_idx_arr[i]]['entity_id']
            s23_id = s23_records[s23_idx_arr[i]]['entity_id']
            all_scored_pairs.append((s1_id, s23_id, preds[i]))
            candidates_dict[s1_id].add(s23_id)
        
        del X, preds
        gc.collect()
    
    # Apply decision layer
    logger.info("=" * 60)
    logger.info("DECISION LAYER")
    logger.info("=" * 60)
    
    predictions = make_predictions(
        all_scored_pairs, all_s1_id_set,
        threshold_first=eval_results['threshold_first'],
        threshold_extra=eval_results['threshold_extra'],
        competition_margin=eval_results['competition_margin'],
    )
    
    # Stats
    n_matched = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    total_matches = sum(len(v) for v in predictions.values())
    
    logger.info(f"  Predicted: {n_matched:,} matched S1, {n_singleton:,} singletons")
    logger.info(f"  Total match links: {total_matches:,}")
    logger.info(f"  Avg matches per matched S1: {total_matches/max(n_matched, 1):.2f}")
    
    # Write output
    logger.info("=" * 60)
    logger.info("WRITING OUTPUT")
    logger.info("=" * 60)
    
    write_output(predictions, candidates_dict, OUTPUT_DIR, all_s1_ids)
    
    elapsed = time.time() - t_start
    logger.info(f"\n  Prediction complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument('--mode', choices=['eda', 'train', 'predict', 'all'],
                       default='all', help='Pipeline mode')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N S1 entities for quick testing')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching of intermediate results')
    
    args = parser.parse_args()
    
    if args.mode == 'eda':
        from eda import run_eda
        run_eda()
    elif args.mode == 'train':
        run_train(args)
    elif args.mode == 'predict':
        run_predict(args)
    elif args.mode == 'all':
        run_train(args)
        run_predict(args)


if __name__ == "__main__":
    main()
