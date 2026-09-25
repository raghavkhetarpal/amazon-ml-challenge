#!/usr/bin/env python3
"""
Production Test Prediction Engine for Business Entity Resolution.

High-throughput, memory-safe (sub-3GB RAM) country-by-country pipeline:
- Fast vectorized record normalization (>40k rec/s)
- Fast word-level TF-IDF (1-2 grams) on name + address with stopword/frequency pruning
- Pre-transposed sparse CSR dot product (1.1 ms per query)
- Direct pre-allocated numpy matrix for batch features
- LightGBM inference + one-to-one competition (precision protection for F0.5)
- Exact row-order matching with test_source1.tsv
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
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SRC_DIR, "../../.."))
sys.path.insert(0, SRC_DIR)

from normalize import normalize_record
from blocking import sparse_cosine_topk
from features import compute_pair_features
from decide import one_to_one_competition

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

TEST_DIR = os.path.join(ROOT, "dataset/test")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_values=[])


def load_country_s23(country):
    """Load S2 and S3 for a single country only to keep memory lean."""
    logger.info(f"  Loading S2 and S3 for {country}...")
    t0 = time.time()
    s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str,
                     quoting=csv.QUOTE_NONE, keep_default_na=False)
    s2_c = s2[s2['country'] == country].copy()
    del s2
    
    s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str,
                     quoting=csv.QUOTE_NONE, keep_default_na=False)
    s3_c = s3[s3['country'] == country].copy()
    del s3
    
    s23_c = pd.concat([s2_c, s3_c], ignore_index=True)
    del s2_c, s3_c
    gc.collect()
    logger.info(f"  Loaded {len(s23_c):,} records for {country} in {time.time()-t0:.1f}s")
    return s23_c


def fast_normalize_df(df, label=""):
    """Normalize dataframe records using numpy array access (40k+ rec/s)."""
    t0 = time.time()
    n = len(df)
    eids = df['entity_id'].values
    names = df['business_name'].values
    addrs = df['business_address'].values
    countries = df['country'].values
    
    records = [None] * n
    for i in range(n):
        records[i] = normalize_record(eids[i], names[i], addrs[i], countries[i])
        
    logger.info(f"    Normalized {n:,} {label} records in {time.time()-t0:.1f}s ({n/max(time.time()-t0, 0.01):.0f} rec/s)")
    return records


def build_country_blocking_models(s23_records, max_features=40000):
    """Fit fast word-level TF-IDF models and build pre-transposed corpus indices."""
    logger.info("    Building fast word-level TF-IDF models & corpus index...")
    t0 = time.time()
    
    s23_names = [r.get('name_core', '') or '' for r in s23_records]
    s23_addrs = [r.get('addr_core', '') or '' for r in s23_records]
    
    # 1. Name Word TF-IDF (1-2 grams)
    name_vec = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        max_features=max_features, sublinear_tf=True, norm='l2',
        min_df=2, max_df=0.4
    )
    s23_name_mat = name_vec.fit_transform(s23_names)
    s23_name_t = s23_name_mat.T.tocsc()
    del s23_name_mat  # Free memory
    
    # 2. Address Word TF-IDF (1-2 grams)
    addr_vec = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        max_features=max_features, sublinear_tf=True, norm='l2',
        min_df=2, max_df=0.25
    )
    s23_addr_mat = addr_vec.fit_transform(s23_addrs)
    s23_addr_t = s23_addr_mat.T.tocsc()
    del s23_addr_mat  # Free memory
    
    gc.collect()
    logger.info(f"    Corpus indexed & pre-transposed in {time.time()-t0:.1f}s: name shape {s23_name_t.shape}, addr shape {s23_addr_t.shape}")
    
    return {
        'name_vec': name_vec,
        's23_name_t': s23_name_t,
        'addr_vec': addr_vec,
        's23_addr_t': s23_addr_t,
    }


def predict_country(country, s1_country_df, s23_country_df, model, feature_names,
                    threshold_first=0.90, threshold_extra=0.90, margin=0.05,
                    top_k_per_channel=10, top_k_final=10, batch_size=5000):
    """Run full blocking, feature extraction, scoring, and competition for one country."""
    logger.info("=" * 70)
    logger.info(f"PROCESSING COUNTRY: {country} ({len(s1_country_df):,} S1, {len(s23_country_df):,} S2/S3)")
    logger.info("=" * 70)
    
    t_start = time.time()
    
    # 1. Normalize
    s1_records = fast_normalize_df(s1_country_df, f"S1 {country}")
    s23_records = fast_normalize_df(s23_country_df, f"S2/S3 {country}")
    
    s1_ids = [r['entity_id'] for r in s1_records]
    s23_ids = [r['entity_id'] for r in s23_records]
    
    # 2. Build index
    models = build_country_blocking_models(s23_records)
    
    # 3. Process S1 in streaming batches
    logger.info(f"  Running blocking, features, and model scoring in batches of {batch_size:,}...")
    
    n_s1 = len(s1_records)
    candidate_dict = {}       # s1_id -> comma-separated string
    high_scoring_pairs = []   # (s1_id, s23_id, score) for competition
    feat_map = {f: i for i, f in enumerate(feature_names)}
    
    for b_start in range(0, n_s1, batch_size):
        b_end = min(b_start + batch_size, n_s1)
        b_t0 = time.time()
        
        batch_s1_records = s1_records[b_start:b_end]
        batch_s1_names = [r.get('name_core', '') or '' for r in batch_s1_records]
        batch_s1_addrs = [r.get('addr_core', '') or '' for r in batch_s1_records]
        
        # Channel 1: Name Word TF-IDF
        q_name = models['name_vec'].transform(batch_s1_names)
        res_name = sparse_cosine_topk(q_name, corpus_t=models['s23_name_t'],
                                      top_k=top_k_per_channel, batch_size=500, min_score=0.15)
        
        # Channel 2: Address Word TF-IDF
        q_addr = models['addr_vec'].transform(batch_s1_addrs)
        res_addr = sparse_cosine_topk(q_addr, corpus_t=models['s23_addr_t'],
                                      top_k=10, batch_size=500, min_score=0.18)
        
        # Fuse candidates for batch
        batch_candidates = {}
        for local_idx in range(len(batch_s1_records)):
            cands = {}
            for c_idx, s in res_name[local_idx]:
                cands[c_idx] = max(cands.get(c_idx, 0.0), s)
            for c_idx, s in res_addr[local_idx]:
                cands[c_idx] = max(cands.get(c_idx, 0.0), s * 0.85)
            
            if cands:
                batch_candidates[local_idx] = sorted(cands.items(), key=lambda x: -x[1])[:top_k_final]
            else:
                batch_candidates[local_idx] = []

        # Direct pre-allocated numpy matrix for batch
        n_batch_pairs = sum(len(c) for c in batch_candidates.values())
        X_batch = np.zeros((n_batch_pairs, len(feature_names)), dtype=np.float32) if n_batch_pairs > 0 else None
        pair_meta = [None] * n_batch_pairs
        pair_idx = 0
        
        for local_idx, s1_rec in enumerate(batch_s1_records):
            sorted_c = batch_candidates[local_idx]
            global_s1_idx = b_start + local_idx
            s1_id = s1_ids[global_s1_idx]
            
            if sorted_c:
                cand_eids = [s23_ids[c_idx] for c_idx, _ in sorted_c]
                candidate_dict[s1_id] = ",".join(cand_eids)
                
                n_cands = len(sorted_c)
                top1_s = sorted_c[0][1] if n_cands > 0 else 0
                top2_s = sorted_c[1][1] if n_cands > 1 else 0
                
                for rank, (c_idx, block_score) in enumerate(sorted_c):
                    s23_rec = s23_records[c_idx]
                    feats = compute_pair_features(
                        s1_rec, s23_rec,
                        blocking_score=block_score,
                        candidate_rank=rank,
                        n_candidates=n_cands,
                        top1_score=top1_s,
                        top2_score=top2_s
                    )
                    for fname, fval in feats.items():
                        if fname in feat_map:
                            X_batch[pair_idx, feat_map[fname]] = fval
                    pair_meta[pair_idx] = (global_s1_idx, c_idx)
                    pair_idx += 1
            else:
                candidate_dict[s1_id] = ""
        
        # Batch prediction
        if X_batch is not None and n_batch_pairs > 0:
            preds = model.predict(X_batch)
            for p_idx in range(n_batch_pairs):
                score = preds[p_idx]
                if score >= 0.65:
                    g_s1_idx, c_idx = pair_meta[p_idx]
                    high_scoring_pairs.append((s1_ids[g_s1_idx], s23_ids[c_idx], float(score)))
            del X_batch, preds, pair_meta
            
        del q_name, q_addr, res_name, res_addr, batch_candidates
        logger.info(f"     Batch {b_start:,}-{b_end:,} ({100*b_end/n_s1:.1f}%) in {time.time()-b_t0:.1f}s | Pairs: {n_batch_pairs:,} | High-score pairs accumulated: {len(high_scoring_pairs):,}")
        sys.stdout.flush()
    
    # 4. One-to-one competition & decision layer
    logger.info(f"  Running one-to-one competition on {len(high_scoring_pairs):,} pairs (margin={margin})...")
    competed_pairs = one_to_one_competition(high_scoring_pairs, margin=margin)
    
    # Group by S1
    s1_matches = defaultdict(list)
    for s1_id, s23_id, score in competed_pairs:
        s1_matches[s1_id].append((s23_id, score))
        
    matches_dict = {}
    n_singletons = 0
    total_matched_links = 0
    
    for s1_id in s1_ids:
        matches = s1_matches.get(s1_id, [])
        if not matches:
            matches_dict[s1_id] = ""
            n_singletons += 1
            continue
            
        matches.sort(key=lambda x: -x[1])
        if matches[0][1] < threshold_first:
            matches_dict[s1_id] = ""
            n_singletons += 1
            continue
            
        accepted = [matches[0][0]]
        for s23_id, score in matches[1:]:
            if score >= threshold_extra:
                accepted.append(s23_id)
                
        matches_dict[s1_id] = ",".join(sorted(accepted))
        total_matched_links += len(accepted)
        
    logger.info(f"  Country {country} complete in {time.time()-t_start:.1f}s:")
    logger.info(f"     Total S1 entities:      {n_s1:,}")
    logger.info(f"     Predicted singletons:   {n_singletons:,} ({100*n_singletons/n_s1:.2f}%)")
    logger.info(f"     Non-singleton entities: {n_s1 - n_singletons:,} ({100*(n_s1 - n_singletons)/n_s1:.2f}%)")
    logger.info(f"     Total match links:      {total_matched_links:,}")
    logger.info(f"     Avg links per matched:  {total_matched_links/max(n_s1 - n_singletons, 1):.2f}")
    
    # Clean memory
    del models, s1_records, s23_records, high_scoring_pairs, competed_pairs
    gc.collect()
    
    return candidate_dict, matches_dict


def check_country_checkpoint(country, expected_count):
    cand_file = os.path.join(OUTPUT_DIR, f"candidates_{country}.tsv")
    match_file = os.path.join(OUTPUT_DIR, f"matches_{country}.tsv")
    if not (os.path.isfile(cand_file) and os.path.isfile(match_file)):
        return False, None, None
    
    def count_lines(fname):
        with open(fname, 'rb') as f:
            return sum(1 for _ in f)
            
    c_lines = count_lines(cand_file)
    m_lines = count_lines(match_file)
    
    if c_lines == expected_count + 1 and m_lines == expected_count + 1:
        logger.info(f"  Valid checkpoint found for {country} ({expected_count:,} rows). Loading...")
        cand_dict = {}
        with open(cand_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if row:
                    cand_dict[row[0]] = row[1] if len(row) > 1 else ""
        match_dict = {}
        with open(match_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if row:
                    match_dict[row[0]] = row[1] if len(row) > 1 else ""
        logger.info(f"  Successfully loaded {len(match_dict):,} records for {country} from disk.")
        return True, cand_dict, match_dict
    else:
        logger.warning(f"  Incomplete checkpoint for {country}: cand lines={c_lines}, match lines={m_lines}, expected {expected_count+1}")
        return False, None, None


def main():
    parser = argparse.ArgumentParser(description="Generate Final Test Submission")
    parser.add_argument('--sample', type=int, default=None, help="Sample N S1 entities per country for fast validation")
    parser.add_argument('--batch-size', type=int, default=5000, help="S1 batch size for streaming")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PRODUCTION TEST SUBMISSION GENERATION")
    logger.info("=" * 80)
    
    t_global_start = time.time()
    
    # Load model
    model_path = os.path.join(ARTIFACTS_DIR, "final_model.pkl")
    logger.info(f"Loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    model = saved['model']
    feature_names = saved['feature_names']
    eval_results = saved.get('eval_results', {})
    
    t_first = eval_results.get('threshold_first', 0.90)
    t_extra = eval_results.get('threshold_extra', 0.90)
    margin = eval_results.get('competition_margin', 0.05)
    logger.info(f"Model parameters: T_first={t_first}, T_extra={t_extra}, margin={margin}")
    
    # Load test source 1
    logger.info("Loading test source 1...")
    test_s1 = load_tsv(os.path.join(TEST_DIR, "test_source1.tsv"))
    all_test_s1_ids = test_s1['entity_id'].tolist()
    logger.info(f"Test S1 entities: {len(test_s1):,}")
    
    # Process country-by-country (loading only that country's S2/S3 records into memory)
    countries = ['France', 'US', 'India']
    
    all_candidates = {}
    all_matches = {}
    
    for country in countries:
        s1_c = test_s1[test_s1['country'] == country].copy()
        
        if args.sample and len(s1_c) > args.sample:
            logger.info(f"  Sampling {args.sample} S1 for {country} (test smoke run)...")
            s1_c = s1_c.head(args.sample)
            
        n_expected = len(s1_c)
        has_cp, c_cand, c_match = check_country_checkpoint(country, n_expected)
        
        if has_cp:
            logger.info(f"  Skipping recomputation for {country} (loaded from disk checkpoint).")
        else:
            s23_c = load_country_s23(country)
            
            c_cand, c_match = predict_country(
                country, s1_c, s23_c, model, feature_names,
                threshold_first=t_first, threshold_extra=t_extra, margin=margin,
                batch_size=args.batch_size
            )
            del s23_c
            gc.collect()
            
            # Immediately persist country checkpoint to disk
            cand_file = os.path.join(OUTPUT_DIR, f"candidates_{country}.tsv")
            match_file = os.path.join(OUTPUT_DIR, f"matches_{country}.tsv")
            
            logger.info(f"  Saving checkpoint for {country} to {cand_file} and {match_file}...")
            with open(cand_file, 'w', encoding='utf-8', newline='') as f:
                f.write("source1_entity_id\tcandidate_entity_ids\n")
                for s1_id in s1_c['entity_id'].values:
                    f.write(f"{s1_id}\t{c_cand.get(s1_id, '')}\n")
                    
            with open(match_file, 'w', encoding='utf-8', newline='') as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                for s1_id in s1_c['entity_id'].values:
                    f.write(f"{s1_id}\t{c_match.get(s1_id, '')}\n")
            logger.info(f"  Checkpoint for {country} saved successfully.")
            
        del s1_c
        gc.collect()
        
        all_candidates.update(c_cand)
        all_matches.update(c_match)
        
    logger.info("=" * 80)
    logger.info("WRITING OUTPUT FILES (Exact S1 Order Preserved)")
    logger.info("=" * 80)
    
    # 1. matching_results.tsv
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    logger.info(f"Writing {matching_file}...")
    with open(matching_file, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1_ids:
            if args.sample and s1_id not in all_matches:
                continue
            matched_str = all_matches.get(s1_id, "")
            f.write(f"{s1_id}\t{matched_str}\n")
            
    # 2. candidate_pairs.tsv
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    logger.info(f"Writing {candidate_file}...")
    with open(candidate_file, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1_ids:
            if args.sample and s1_id not in all_candidates:
                continue
            cands_str = all_candidates.get(s1_id, "")
            f.write(f"{s1_id}\t{cands_str}\n")
            
    logger.info(f"Production pipeline finished in {time.time()-t_global_start:.1f}s ({((time.time()-t_global_start)/60):.1f} min)")
    logger.info(f"  Matching results: {matching_file}")
    logger.info(f"  Candidate pairs:  {candidate_file}")


if __name__ == "__main__":
    main()
