#!/usr/bin/env python3
"""
Leave-One-Country-Out (LOCO) Experiment.
Evaluates cross-country generalization to simulate the unseen France test set:
1. Train on US -> Evaluate on India
2. Train on India -> Evaluate on US
Reports Macro-F0.5, Singleton F0.5, Non-singleton F0.5, and Generalization Gap.
"""

import argparse
import csv
import gc
import logging
import os
import sys
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, "../../.."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_record
from blocking import BlockingEngine
from run_pipeline import compute_all_features
from metric import macro_f05, macro_f05_detailed
from decide import make_predictions, tune_thresholds

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TRAIN_DIR = os.path.join(ROOT, "dataset/train")
SEED = 42


def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_values=[])


def load_ground_truth(path):
    df = load_tsv(path)
    gt = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids'].strip()
        gt[s1_id] = set(matched.split(',')) if matched else set()
    return gt


def run_country_pipeline(s1_df_country, s23_df_country, gt, sample_s1=5000):
    """Normalize, block, compute features, and prepare training/eval data for one country."""
    np.random.seed(SEED)
    if len(s1_df_country) > sample_s1:
        s1_sample = s1_df_country.sample(n=sample_s1, random_state=SEED).reset_index(drop=True)
    else:
        s1_sample = s1_df_country.reset_index(drop=True)
    
    s1_ids_set = set(s1_sample['entity_id'])
    
    # Ground truth for this sample
    gt_sample = {k: v for k, v in gt.items() if k in s1_ids_set}
    
    # Relevant S2/S3
    matched_s23 = set()
    for mids in gt_sample.values():
        matched_s23.update(mids)
    
    s23_matched_df = s23_df_country[s23_df_country['entity_id'].isin(matched_s23)]
    s23_unmatched_df = s23_df_country[~s23_df_country['entity_id'].isin(matched_s23)]
    if len(s23_unmatched_df) > sample_s1 * 15:
        s23_unmatched_df = s23_unmatched_df.sample(n=sample_s1 * 15, random_state=SEED)
    
    s23_sample = pd.concat([s23_matched_df, s23_unmatched_df], ignore_index=True).drop_duplicates(subset=['entity_id']).reset_index(drop=True)
    
    logger.info(f"  Country dataset: {len(s1_sample):,} S1, {len(s23_sample):,} S2/S3")
    
    # Normalize
    s1_records = [normalize_record(row['entity_id'], row['business_name'], row['business_address'], row['country']) for _, row in s1_sample.iterrows()]
    s23_records = [normalize_record(row['entity_id'], row['business_name'], row['business_address'], row['country']) for _, row in s23_sample.iterrows()]
    
    # Block
    blocker = BlockingEngine(top_k_per_channel=40, top_k_final=80)
    candidates = blocker.generate_candidates(s1_records, s23_records, country=s1_sample['country'].iloc[0])
    
    s1_ids = [r['entity_id'] for r in s1_records]
    s23_ids = [r['entity_id'] for r in s23_records]
    blocking_eval = blocker.evaluate_blocking_recall(candidates, s1_records, s23_records, gt_sample, s1_ids, s23_ids)
    logger.info(f"  Blocking recall: {blocking_eval['recall']:.4f} ({blocking_eval['found_pairs']:,}/{blocking_eval['total_true_pairs']:,})")
    
    # Features
    s1_idx_arr, s23_idx_arr, X, feature_names = compute_all_features(s1_records, s23_records, candidates)
    
    # Labels
    y = np.zeros(len(s1_idx_arr), dtype=np.float32)
    for i in range(len(s1_idx_arr)):
        s1_id = s1_ids[s1_idx_arr[i]]
        s23_id = s23_ids[s23_idx_arr[i]]
        if s23_id in gt_sample.get(s1_id, set()):
            y[i] = 1.0
            
    logger.info(f"  Features: {X.shape[0]:,} pairs x {X.shape[1]} feats, {int(y.sum()):,} positives ({y.mean()*100:.2f}%)")
    
    return {
        's1_sample': s1_sample,
        's23_sample': s23_sample,
        's1_records': s1_records,
        's23_records': s23_records,
        's1_ids': s1_ids,
        's23_ids': s23_ids,
        's1_ids_set': s1_ids_set,
        'gt_sample': gt_sample,
        'candidates': candidates,
        's1_idx_arr': s1_idx_arr,
        's23_idx_arr': s23_idx_arr,
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'blocking_recall': blocking_eval['recall']
    }


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Country-Out Evaluation")
    parser.add_argument('--sample', type=int, default=4000, help="S1 sample size per country")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("LEAVE-ONE-COUNTRY-OUT (LOCO) GENERALIZATION EXPERIMENT")
    logger.info("=" * 70)
    
    # Load raw data
    logger.info("Loading training data...")
    s1_df = load_tsv(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    s2_df = load_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"))
    s3_df = load_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    # Split by country
    logger.info("Preparing US dataset...")
    us_data = run_country_pipeline(s1_df[s1_df['country'] == 'US'], s23_df[s23_df['country'] == 'US'], gt, sample_s1=args.sample)
    
    logger.info("Preparing India dataset...")
    in_data = run_country_pipeline(s1_df[s1_df['country'] == 'India'], s23_df[s23_df['country'] == 'India'], gt, sample_s1=args.sample)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
    }
    
    # -------------------------------------------------------------
    # Experiment 1: Train on US -> Test on India (Simulating shift)
    # -------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1: Train on US -> Test on India")
    logger.info("=" * 60)
    
    us_train_data = lgb.Dataset(us_data['X'], label=us_data['y'], feature_name=us_data['feature_names'])
    model_us = lgb.train(params, us_train_data, num_boost_round=120)
    
    # Predict on India
    in_preds = model_us.predict(in_data['X'])
    in_scored = []
    for i in range(len(in_preds)):
        s1_id = in_data['s1_ids'][in_data['s1_idx_arr'][i]]
        s23_id = in_data['s23_ids'][in_data['s23_idx_arr'][i]]
        in_scored.append((s1_id, s23_id, in_preds[i]))
        
    in_decisions = make_predictions(in_scored, in_data['s1_ids_set'], threshold_first=0.85, threshold_extra=0.85, competition_margin=0.05)
    score_us_to_in = macro_f05_detailed(in_decisions, in_data['gt_sample'])
    
    logger.info(f"Train on US -> India Test Results:")
    logger.info(f"  Overall Macro-F0.5: {score_us_to_in['overall']:.4f}")
    logger.info(f"  Singleton F0.5:     {score_us_to_in['singleton']:.4f}")
    logger.info(f"  Non-singleton F0.5: {score_us_to_in['non_singleton']:.4f}")
    
    # -------------------------------------------------------------
    # Experiment 2: Train on India -> Test on US
    # -------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: Train on India -> Test on US")
    logger.info("=" * 60)
    
    in_train_data = lgb.Dataset(in_data['X'], label=in_data['y'], feature_name=in_data['feature_names'])
    model_in = lgb.train(params, in_train_data, num_boost_round=120)
    
    # Predict on US
    us_preds = model_in.predict(us_data['X'])
    us_scored = []
    for i in range(len(us_preds)):
        s1_id = us_data['s1_ids'][us_data['s1_idx_arr'][i]]
        s23_id = us_data['s23_ids'][us_data['s23_idx_arr'][i]]
        us_scored.append((s1_id, s23_id, us_preds[i]))
        
    us_decisions = make_predictions(us_scored, us_data['s1_ids_set'], threshold_first=0.85, threshold_extra=0.85, competition_margin=0.05)
    score_in_to_us = macro_f05_detailed(us_decisions, us_data['gt_sample'])
    
    logger.info(f"Train on India -> US Test Results:")
    logger.info(f"  Overall Macro-F0.5: {score_in_to_us['overall']:.4f}")
    logger.info(f"  Singleton F0.5:     {score_in_to_us['singleton']:.4f}")
    logger.info(f"  Non-singleton F0.5: {score_in_to_us['non_singleton']:.4f}")
    
    # -------------------------------------------------------------
    # In-domain baselines for reference
    # -------------------------------------------------------------
    # US in-domain
    us_self_preds = model_us.predict(us_data['X'])
    us_self_scored = [(us_data['s1_ids'][us_data['s1_idx_arr'][i]], us_data['s23_ids'][us_data['s23_idx_arr'][i]], us_self_preds[i]) for i in range(len(us_self_preds))]
    score_us_in_domain = macro_f05_detailed(make_predictions(us_self_scored, us_data['s1_ids_set'], threshold_first=0.85, threshold_extra=0.85, competition_margin=0.05), us_data['gt_sample'])
    
    # India in-domain
    in_self_preds = model_in.predict(in_data['X'])
    in_self_scored = [(in_data['s1_ids'][in_data['s1_idx_arr'][i]], in_data['s23_ids'][in_data['s23_idx_arr'][i]], in_self_preds[i]) for i in range(len(in_self_preds))]
    score_in_in_domain = macro_f05_detailed(make_predictions(in_self_scored, in_data['s1_ids_set'], threshold_first=0.85, threshold_extra=0.85, competition_margin=0.05), in_data['gt_sample'])
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: LEAVE-ONE-COUNTRY-OUT GENERALIZATION")
    logger.info("=" * 70)
    gap_us = score_us_in_domain['overall'] - score_in_to_us['overall']
    gap_in = score_in_in_domain['overall'] - score_us_to_in['overall']
    logger.info(f"US:    In-domain F0.5 = {score_us_in_domain['overall']:.4f} | From India F0.5 = {score_in_to_us['overall']:.4f} (Gap: {gap_us:+.4f})")
    logger.info(f"India: In-domain F0.5 = {score_in_in_domain['overall']:.4f} | From US F0.5    = {score_us_to_in['overall']:.4f} (Gap: {gap_in:+.4f})")
    logger.info(f"Average cross-country transfer gap: {(gap_us + gap_in)/2:.4f}")
    
    return {
        'score_us_to_in': score_us_to_in,
        'score_in_to_us': score_in_to_us,
        'score_us_in_domain': score_us_in_domain,
        'score_in_in_domain': score_in_in_domain,
        'gap_us': gap_us,
        'gap_in': gap_in,
    }


if __name__ == "__main__":
    main()
