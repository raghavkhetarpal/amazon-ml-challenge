#!/usr/bin/env python3
"""
Stage 3: Feature engineering for candidate pairs.

Computes pairwise features between S1 and S2/S3 records:
- Name similarity: Jaccard, char-ngram cosine, Jaro-Winkler, Levenshtein, etc.
- Address similarity: token overlap, postal code match, house number match
- Context features: candidate rank, score gaps, source indicator
"""

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler


def _jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _token_jaccard(text_a, text_b):
    """Token-level Jaccard similarity."""
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    return _jaccard(tokens_a, tokens_b)


def _char_ngram_jaccard(text_a, text_b, n=3):
    """Character n-gram Jaccard similarity."""
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    ngrams_a = {text_a[i:i+n] for i in range(max(0, len(text_a)-n+1))}
    ngrams_b = {text_b[i:i+n] for i in range(max(0, len(text_b)-n+1))}
    return _jaccard(ngrams_a, ngrams_b)


def _containment(text_a, text_b):
    """Fraction of tokens in shorter text that appear in longer text."""
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    shorter = tokens_a if len(tokens_a) <= len(tokens_b) else tokens_b
    longer = tokens_b if len(tokens_a) <= len(tokens_b) else tokens_a
    if not shorter:
        return 0.0
    return len(shorter & longer) / len(shorter)


def _length_ratio(text_a, text_b):
    """Length ratio (min/max)."""
    if not text_a and not text_b:
        return 1.0
    la = len(text_a) if text_a else 0
    lb = len(text_b) if text_b else 0
    if max(la, lb) == 0:
        return 1.0
    return min(la, lb) / max(la, lb)


def compute_pair_features(s1_rec, s23_rec, blocking_score=0.0, candidate_rank=0,
                           n_candidates=0, top1_score=0.0, top2_score=0.0):
    """Compute features for a single (S1, S2/S3) candidate pair.
    
    Args:
        s1_rec: dict with normalized S1 record fields
        s23_rec: dict with normalized S2/S3 record fields
        blocking_score: score from blocking channel
        candidate_rank: rank within S1's candidate list
        n_candidates: total candidates for this S1
        top1_score: highest blocking score for this S1
        top2_score: 2nd highest blocking score for this S1
        
    Returns:
        dict of feature name -> value
    """
    features = {}
    
    # ---- Name features ----
    name_core_1 = s1_rec.get('name_core', '') or ''
    name_core_2 = s23_rec.get('name_core', '') or ''
    name_norm_1 = s1_rec.get('name_norm', '') or ''
    name_norm_2 = s23_rec.get('name_norm', '') or ''
    
    # Token Jaccard on core name
    features['name_token_jaccard'] = _token_jaccard(name_core_1, name_core_2)
    
    # Sorted-token Jaccard (transposition robust)
    sorted_1 = s1_rec.get('name_sorted', '') or ''
    sorted_2 = s23_rec.get('name_sorted', '') or ''
    features['name_sorted_jaccard'] = _token_jaccard(sorted_1, sorted_2)
    
    # Char 3-gram Jaccard
    features['name_char3_jaccard'] = _char_ngram_jaccard(name_core_1, name_core_2, 3)
    
    # RapidFuzz scores (0-100 scale, normalize to 0-1)
    features['name_ratio'] = fuzz.ratio(name_core_1, name_core_2) / 100.0
    features['name_partial_ratio'] = fuzz.partial_ratio(name_core_1, name_core_2) / 100.0
    features['name_token_sort_ratio'] = fuzz.token_sort_ratio(name_core_1, name_core_2) / 100.0
    features['name_token_set_ratio'] = fuzz.token_set_ratio(name_core_1, name_core_2) / 100.0
    
    # Jaro-Winkler
    features['name_jaro_winkler'] = JaroWinkler.similarity(name_core_1, name_core_2)
    
    # Normalized Levenshtein
    max_len = max(len(name_core_1), len(name_core_2))
    if max_len > 0:
        features['name_levenshtein_norm'] = 1.0 - Levenshtein.distance(name_core_1, name_core_2) / max_len
    else:
        features['name_levenshtein_norm'] = 1.0
    
    # Containment
    features['name_containment'] = _containment(name_core_1, name_core_2)
    
    # Length ratio
    features['name_length_ratio'] = _length_ratio(name_core_1, name_core_2)
    
    # First token equality
    ft1 = s1_rec.get('name_first_token', '') or ''
    ft2 = s23_rec.get('name_first_token', '') or ''
    features['name_first_token_eq'] = 1.0 if ft1 and ft2 and ft1 == ft2 else 0.0
    
    # Phonetic key equality
    ph1 = s1_rec.get('name_phonetic', '') or ''
    ph2 = s23_rec.get('name_phonetic', '') or ''
    features['name_phonetic_eq'] = 1.0 if ph1 and ph2 and ph1 == ph2 else 0.0
    
    # Legal form compatibility
    lf1 = s1_rec.get('legal_form', '') or ''
    lf2 = s23_rec.get('legal_form', '') or ''
    features['legal_form_eq'] = 1.0 if lf1 == lf2 else 0.0
    features['legal_form_conflict'] = 1.0 if lf1 and lf2 and lf1 != lf2 else 0.0
    features['legal_form_both_empty'] = 1.0 if not lf1 and not lf2 else 0.0
    
    # Name token counts
    n_tokens_1 = len(name_core_1.split()) if name_core_1 else 0
    n_tokens_2 = len(name_core_2.split()) if name_core_2 else 0
    features['name_token_count_diff'] = abs(n_tokens_1 - n_tokens_2)
    features['name_token_count_ratio'] = min(n_tokens_1, n_tokens_2) / max(n_tokens_1, n_tokens_2) if max(n_tokens_1, n_tokens_2) > 0 else 1.0
    
    # ---- Address features ----
    addr_core_1 = s1_rec.get('addr_core', '') or ''
    addr_core_2 = s23_rec.get('addr_core', '') or ''
    
    # Address token Jaccard
    features['addr_token_jaccard'] = _token_jaccard(addr_core_1, addr_core_2)
    
    # Address char 3-gram Jaccard
    features['addr_char3_jaccard'] = _char_ngram_jaccard(addr_core_1, addr_core_2, 3)
    
    # Address fuzzy ratio
    features['addr_ratio'] = fuzz.ratio(addr_core_1, addr_core_2) / 100.0
    features['addr_partial_ratio'] = fuzz.partial_ratio(addr_core_1, addr_core_2) / 100.0
    features['addr_token_sort_ratio'] = fuzz.token_sort_ratio(addr_core_1, addr_core_2) / 100.0
    
    # Postal code match
    pc1 = s1_rec.get('postal_code', '') or ''
    pc2 = s23_rec.get('postal_code', '') or ''
    features['postal_exact'] = 1.0 if pc1 and pc2 and pc1 == pc2 else 0.0
    features['postal_prefix3'] = 1.0 if pc1 and pc2 and len(pc1) >= 3 and len(pc2) >= 3 and pc1[:3] == pc2[:3] else 0.0
    features['postal_conflict'] = 1.0 if pc1 and pc2 and pc1 != pc2 else 0.0
    features['postal_both_missing'] = 1.0 if not pc1 and not pc2 else 0.0
    features['postal_one_missing'] = 1.0 if bool(pc1) != bool(pc2) else 0.0
    
    # House number match
    hn1 = s1_rec.get('house_number', '') or ''
    hn2 = s23_rec.get('house_number', '') or ''
    features['house_number_eq'] = 1.0 if hn1 and hn2 and hn1 == hn2 else 0.0
    features['house_number_conflict'] = 1.0 if hn1 and hn2 and hn1 != hn2 else 0.0
    
    # Address missing flags
    features['addr_s1_empty'] = 1.0 if not addr_core_1 else 0.0
    features['addr_s23_empty'] = 1.0 if not addr_core_2 else 0.0
    features['addr_both_empty'] = 1.0 if not addr_core_1 and not addr_core_2 else 0.0
    
    # Address length ratio
    features['addr_length_ratio'] = _length_ratio(addr_core_1, addr_core_2)
    
    # ---- Context features ----
    features['blocking_score'] = blocking_score
    features['candidate_rank'] = candidate_rank
    features['n_candidates'] = n_candidates
    features['score_gap_top1_top2'] = top1_score - top2_score
    
    # Source indicator (S2 vs S3)
    eid = s23_rec.get('entity_id', '')
    features['is_s2'] = 1.0 if eid.startswith('S2-') else 0.0
    features['is_s3'] = 1.0 if eid.startswith('S3-') else 0.0
    
    # Name empty flags
    features['name_s1_empty'] = 1.0 if not name_core_1 else 0.0
    features['name_s23_empty'] = 1.0 if not name_core_2 else 0.0
    
    return features


def compute_features_batch(s1_records, s23_records, candidates):
    """Compute features for all candidate pairs.
    
    Args:
        s1_records: list of normalized S1 record dicts
        s23_records: list of normalized S2/S3 record dicts
        candidates: dict {s1_idx: [(s23_idx, score), ...]}
    
    Returns:
        list of (s1_idx, s23_idx, feature_dict) tuples
    """
    results = []
    
    for s1_idx, cand_list in candidates.items():
        if not cand_list:
            continue
            
        s1_rec = s1_records[s1_idx]
        n_cands = len(cand_list)
        
        # Get top scores for gap features
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
            results.append((s1_idx, s23_idx, feats))
    
    return results


# Feature names in order for model input
FEATURE_NAMES = None  # Set on first call

def features_to_array(feature_dicts):
    """Convert list of feature dicts to numpy array.
    
    Returns:
        (X, feature_names) where X is (n_pairs, n_features)
    """
    global FEATURE_NAMES
    if not feature_dicts:
        return np.empty((0, 0)), []
    
    if FEATURE_NAMES is None:
        FEATURE_NAMES = sorted(feature_dicts[0].keys())
    
    X = np.array([[d.get(f, 0.0) for f in FEATURE_NAMES] for d in feature_dicts], dtype=np.float32)
    return X, FEATURE_NAMES


if __name__ == "__main__":
    # Smoke test
    s1 = {'name_core': 'acme corp', 'name_norm': 'acme corp', 'name_sorted': 'acme corp',
           'name_first_token': 'acme', 'name_phonetic': 'akm krb', 'legal_form': 'corp',
           'addr_core': '123 main street new york', 'postal_code': '10001', 
           'house_number': '123', 'entity_id': 'S1-001'}
    s23 = {'name_core': 'acme corporation', 'name_norm': 'acme corporation', 
            'name_sorted': 'acme corporation',
            'name_first_token': 'acme', 'name_phonetic': 'akm krbrtn', 'legal_form': 'corp',
            'addr_core': '123 main st new york ny', 'postal_code': '10001',
            'house_number': '123', 'entity_id': 'S2-001'}
    
    feats = compute_pair_features(s1, s23)
    print(f"Features computed: {len(feats)}")
    for k in sorted(feats.keys()):
        print(f"  {k}: {feats[k]:.4f}")
    
    assert feats['name_first_token_eq'] == 1.0
    assert feats['postal_exact'] == 1.0
    assert feats['house_number_eq'] == 1.0
    assert feats['name_token_jaccard'] > 0.3
    print("\nFeature smoke test PASSED!")
