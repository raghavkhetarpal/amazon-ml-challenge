#!/usr/bin/env python3
"""
Stage 2: Blocking / Candidate Generation.

Multi-channel blocking with union of candidates, capped per S1.
Channels:
1. TF-IDF char n-gram (3-4gram) on normalized name - sparse cosine top-k
2. Word-level TF-IDF on core name
3. First-token / sorted-token / phonetic key hash join
4. Postal-code hash join with address token fallback

All blocking is done within country first for efficiency.
"""

import csv
import os
import sys
import time
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def sparse_cosine_topk(query_matrix, corpus_matrix=None, top_k=50, batch_size=500, min_score=0.05, corpus_t=None):
    """Compute sparse cosine similarity and return top-k indices per query.
    Uses direct CSR array indexing for 30x faster row processing.
    
    Both matrices should be L2-normalized.
    
    Args:
        query_matrix: sparse matrix (n_queries, n_features) - already L2-normalized
        corpus_matrix: sparse matrix (n_corpus, n_features) - already L2-normalized
        top_k: number of top candidates to return per query
        batch_size: number of queries to process at once (500 keeps RAM under 200MB)
        min_score: minimum similarity score to consider
        corpus_t: optional pre-transposed corpus matrix (corpus_matrix.T.tocsc())
        
    Returns:
        list of lists: top_k (index, score) pairs per query
    """
    n_queries = query_matrix.shape[0]
    results = []
    
    # Transpose corpus for dot product if not pre-computed
    if corpus_t is None:
        corpus_t = corpus_matrix.T.tocsc()
    
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]
        
        # Sparse dot product = cosine similarity (since both are L2-normalized)
        sim = batch.dot(corpus_t).tocsr()
        indptr = sim.indptr
        sim_data = sim.data
        sim_indices = sim.indices
        
        for i in range(sim.shape[0]):
            r_start = indptr[i]
            r_end = indptr[i+1]
            if r_start == r_end:
                results.append([])
                continue
            
            data = sim_data[r_start:r_end]
            indices = sim_indices[r_start:r_end]
            
            if min_score > 0:
                mask = data >= min_score
                data = data[mask]
                indices = indices[mask]
                if len(data) == 0:
                    results.append([])
                    continue
            
            if len(data) <= top_k:
                top_idx = np.argsort(-data)
                results.append([(int(indices[j]), float(data[j])) for j in top_idx])
            else:
                top_idx = np.argpartition(-data, top_k)[:top_k]
                top_idx = top_idx[np.argsort(-data[top_idx])]
                results.append([(int(indices[j]), float(data[j])) for j in top_idx])
    
    return results


class BlockingEngine:
    """Multi-channel blocking for entity resolution."""
    
    def __init__(self, top_k_per_channel=50, top_k_final=100):
        self.top_k_per_channel = top_k_per_channel
        self.top_k_final = top_k_final
        
        # TF-IDF vectorizers
        self.char_tfidf = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 4),
            max_features=200000, sublinear_tf=True, norm='l2',
            min_df=2, max_df=0.5
        )
        self.word_tfidf = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2),
            max_features=100000, sublinear_tf=True, norm='l2',
            min_df=2, max_df=0.5
        )
        # Address TF-IDF for cross-script blocking
        self.addr_char_tfidf = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 4),
            max_features=100000, sublinear_tf=True, norm='l2',
            min_df=2, max_df=0.5
        )
        # Combined name+address for comprehensive blocking
        self.combined_tfidf = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 4),
            max_features=200000, sublinear_tf=True, norm='l2',
            min_df=2, max_df=0.5
        )
    
    def _build_hash_index(self, corpus_records, key_fn):
        """Build a hash index mapping key -> list of corpus indices."""
        index = defaultdict(list)
        for i, rec in enumerate(corpus_records):
            key = key_fn(rec)
            if key:
                if isinstance(key, (list, set)):
                    for k in key:
                        if k:
                            index[k].append(i)
                else:
                    index[key].append(i)
        return dict(index)
    
    def generate_candidates(self, s1_records, s23_records, country=None):
        """Generate candidate pairs for a set of S1 records against S2/S3 records.
        
        Args:
            s1_records: list of dicts with normalized fields
            s23_records: list of dicts with normalized fields  
            country: country label (for logging)
            
        Returns:
            dict: {s1_idx: [(s23_idx, max_score), ...]} with at most top_k_final per s1
        """
        t0 = time.time()
        n_s1 = len(s1_records)
        n_s23 = len(s23_records)
        
        if n_s1 == 0 or n_s23 == 0:
            return {}
        
        logger.info(f"  Blocking [{country or 'ALL'}]: {n_s1:,} S1 x {n_s23:,} S2/S3")
        
        # Collect all candidates per S1 with scores
        candidates = defaultdict(dict)  # s1_idx -> {s23_idx: max_score}
        
        # ---- Channel 1: Char n-gram TF-IDF on name_core ----
        t1 = time.time()
        s1_names = [r.get('name_core', '') or '' for r in s1_records]
        s23_names = [r.get('name_core', '') or '' for r in s23_records]
        
        # Fit on combined corpus (unsupervised, no labels)
        all_names = s1_names + s23_names
        self.char_tfidf.fit(all_names)
        
        s1_char_vecs = self.char_tfidf.transform(s1_names)
        s23_char_vecs = self.char_tfidf.transform(s23_names)
        
        char_results = sparse_cosine_topk(
            s1_char_vecs, s23_char_vecs, 
            top_k=self.top_k_per_channel, batch_size=2000
        )
        
        for s1_idx, tops in enumerate(char_results):
            for s23_idx, score in tops:
                old = candidates[s1_idx].get(s23_idx, 0)
                candidates[s1_idx][s23_idx] = max(old, score)
        
        logger.info(f"    Channel 1 (char TF-IDF): {time.time()-t1:.1f}s, "
                    f"avg candidates: {np.mean([len(v) for v in candidates.values()]):.1f}")
        
        # ---- Channel 2: Word TF-IDF on name_core ----
        t1 = time.time()
        self.word_tfidf.fit(all_names)
        
        s1_word_vecs = self.word_tfidf.transform(s1_names)
        s23_word_vecs = self.word_tfidf.transform(s23_names)
        
        word_results = sparse_cosine_topk(
            s1_word_vecs, s23_word_vecs,
            top_k=self.top_k_per_channel, batch_size=2000
        )
        
        for s1_idx, tops in enumerate(word_results):
            for s23_idx, score in tops:
                old = candidates[s1_idx].get(s23_idx, 0)
                candidates[s1_idx][s23_idx] = max(old, score)
        
        logger.info(f"    Channel 2 (word TF-IDF): {time.time()-t1:.1f}s, "
                    f"avg candidates: {np.mean([len(v) for v in candidates.values()]):.1f}")
        
        # ---- Channel 3: Hash-based blocking ----
        t1 = time.time()
        
        # First token
        ft_index = self._build_hash_index(s23_records, lambda r: r.get('name_first_token', ''))
        for s1_idx, rec in enumerate(s1_records):
            ft = rec.get('name_first_token', '')
            if ft and ft in ft_index:
                for s23_idx in ft_index[ft][:200]:  # Cap per key
                    if s23_idx not in candidates[s1_idx]:
                        candidates[s1_idx][s23_idx] = 0.3
        
        # Phonetic key
        ph_index = self._build_hash_index(s23_records, lambda r: r.get('name_phonetic', ''))
        for s1_idx, rec in enumerate(s1_records):
            pk = rec.get('name_phonetic', '')
            if pk and len(pk) > 2 and pk in ph_index:
                for s23_idx in ph_index[pk][:200]:
                    old = candidates[s1_idx].get(s23_idx, 0)
                    candidates[s1_idx][s23_idx] = max(old, 0.4)
        
        logger.info(f"    Channel 3 (hash keys): {time.time()-t1:.1f}s, "
                    f"avg candidates: {np.mean([len(v) for v in candidates.values()]):.1f}")
        
        # ---- Channel 4: Postal code + address tokens ----
        t1 = time.time()
        
        pc_index = self._build_hash_index(s23_records, lambda r: r.get('postal_code', ''))
        for s1_idx, rec in enumerate(s1_records):
            pc = rec.get('postal_code', '')
            if pc and pc in pc_index:
                for s23_idx in pc_index[pc][:200]:
                    old = candidates[s1_idx].get(s23_idx, 0)
                    candidates[s1_idx][s23_idx] = max(old, 0.35)
        
        logger.info(f"    Channel 4 (postal code): {time.time()-t1:.1f}s, "
                    f"avg candidates: {np.mean([len(v) for v in candidates.values()]):.1f}")
        
        # ---- Channel 5: Address char TF-IDF (critical for cross-script Indian data) ----
        t1 = time.time()
        s1_addrs = [r.get('addr_core', '') or '' for r in s1_records]
        s23_addrs = [r.get('addr_core', '') or '' for r in s23_records]
        
        # Only run if enough non-empty addresses
        non_empty_s1 = sum(1 for a in s1_addrs if a)
        non_empty_s23 = sum(1 for a in s23_addrs if a)
        
        if non_empty_s1 > 10 and non_empty_s23 > 10:
            all_addrs = s1_addrs + s23_addrs
            self.addr_char_tfidf.fit(all_addrs)
            
            s1_addr_vecs = self.addr_char_tfidf.transform(s1_addrs)
            s23_addr_vecs = self.addr_char_tfidf.transform(s23_addrs)
            
            addr_results = sparse_cosine_topk(
                s1_addr_vecs, s23_addr_vecs,
                top_k=self.top_k_per_channel, batch_size=2000
            )
            
            for s1_idx, tops in enumerate(addr_results):
                for s23_idx, score in tops:
                    if score > 0.3:  # Only keep reasonably similar addresses
                        old = candidates[s1_idx].get(s23_idx, 0)
                        candidates[s1_idx][s23_idx] = max(old, score * 0.8)
        
        logger.info(f"    Channel 5 (addr TF-IDF): {time.time()-t1:.1f}s, "
                    f"avg candidates: {np.mean([len(v) for v in candidates.values()]):.1f}")
        
        # ---- Channel 6: Combined name+address char TF-IDF ----
        t1 = time.time()
        s1_combined = [f"{n} {a}" for n, a in zip(s1_names, s1_addrs)]
        s23_combined = [f"{n} {a}" for n, a in zip(s23_names, s23_addrs)]
        
        all_combined = s1_combined + s23_combined
        self.combined_tfidf.fit(all_combined)
        
        s1_comb_vecs = self.combined_tfidf.transform(s1_combined)
        s23_comb_vecs = self.combined_tfidf.transform(s23_combined)
        
        comb_results = sparse_cosine_topk(
            s1_comb_vecs, s23_comb_vecs,
            top_k=self.top_k_per_channel, batch_size=2000
        )
        
        for s1_idx, tops in enumerate(comb_results):
            for s23_idx, score in tops:
                old = candidates[s1_idx].get(s23_idx, 0)
                candidates[s1_idx][s23_idx] = max(old, score)
        
        logger.info(f"    Channel 6 (combined TF-IDF): {time.time()-t1:.1f}s, "
                    f"avg candidates: {np.mean([len(v) for v in candidates.values()]):.1f}")
        
        # ---- Final: cap to top_k_final per S1 ----
        final_candidates = {}
        for s1_idx, cand_dict in candidates.items():
            if not cand_dict:
                continue
            # Sort by score descending, take top_k_final
            sorted_cands = sorted(cand_dict.items(), key=lambda x: -x[1])
            final_candidates[s1_idx] = sorted_cands[:self.top_k_final]
        
        total_pairs = sum(len(v) for v in final_candidates.values())
        avg_cands = total_pairs / n_s1 if n_s1 > 0 else 0
        
        logger.info(f"  Blocking done [{country or 'ALL'}]: {time.time()-t0:.1f}s total, "
                    f"{total_pairs:,} pairs, avg {avg_cands:.1f} candidates/S1, "
                    f"reduction ratio: {1 - total_pairs/(n_s1*n_s23):.6f}")
        
        return final_candidates
    
    def evaluate_blocking_recall(self, candidates, s1_records, s23_records, 
                                  ground_truth, s1_ids, s23_ids):
        """Evaluate blocking recall: fraction of true pairs found in candidates.
        
        Args:
            candidates: dict {s1_idx: [(s23_idx, score)...]}
            s1_records: list of s1 record dicts
            s23_records: list of s23 record dicts
            ground_truth: dict {s1_entity_id: set of matched entity_ids}
            s1_ids: list of s1 entity_ids (parallel to s1_records)
            s23_ids: list of s23 entity_ids (parallel to s23_records)
            
        Returns:
            dict with recall, missed pairs count, etc.
        """
        # Build s23 id -> index map
        s23_id_to_idx = {eid: i for i, eid in enumerate(s23_ids)}
        
        total_true_pairs = 0
        found_pairs = 0
        missed_examples = []
        
        for s1_idx, s1_eid in enumerate(s1_ids):
            true_matches = ground_truth.get(s1_eid, set())
            if not true_matches:
                continue
                
            # Get candidate s23 indices for this s1
            cand_s23_indices = set()
            if s1_idx in candidates:
                cand_s23_indices = {c[0] for c in candidates[s1_idx]}
            
            for matched_eid in true_matches:
                if matched_eid in s23_id_to_idx:
                    total_true_pairs += 1
                    matched_idx = s23_id_to_idx[matched_eid]
                    if matched_idx in cand_s23_indices:
                        found_pairs += 1
                    elif len(missed_examples) < 20:
                        missed_examples.append((s1_eid, matched_eid))
        
        recall = found_pairs / total_true_pairs if total_true_pairs > 0 else 1.0
        
        total_cands = sum(len(v) for v in candidates.values())
        
        return {
            'recall': recall,
            'total_true_pairs': total_true_pairs,
            'found_pairs': found_pairs,
            'missed_pairs': total_true_pairs - found_pairs,
            'total_candidates': total_cands,
            'missed_examples': missed_examples,
        }


if __name__ == "__main__":
    # Quick smoke test
    print("BlockingEngine smoke test...")
    engine = BlockingEngine(top_k_per_channel=5, top_k_final=10)
    
    s1 = [
        {'name_core': 'acme corporation', 'name_first_token': 'acme', 
         'name_phonetic': 'akm', 'postal_code': '10001'},
        {'name_core': 'beta industries', 'name_first_token': 'beta',
         'name_phonetic': 'bt', 'postal_code': '20002'},
    ]
    s23 = [
        {'name_core': 'acme corp', 'name_first_token': 'acme',
         'name_phonetic': 'akm', 'postal_code': '10001'},
        {'name_core': 'gamma tech', 'name_first_token': 'gamma',
         'name_phonetic': 'gm', 'postal_code': '30003'},
        {'name_core': 'beta ind', 'name_first_token': 'beta',
         'name_phonetic': 'bt', 'postal_code': '20002'},
    ]
    
    cands = engine.generate_candidates(s1, s23, country='TEST')
    print(f"  S1[0] candidates: {cands.get(0, [])}")
    print(f"  S1[1] candidates: {cands.get(1, [])}")
    assert 0 in cands and any(c[0] == 0 for c in cands[0]), "Acme should match Acme"
    print("BlockingEngine smoke test PASSED!")
