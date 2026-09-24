#!/usr/bin/env python3
"""
Stage 4: Decision layer for entity resolution.

Applies threshold-based decisions and one-to-one competition
to convert model scores into final matches.
"""

from collections import defaultdict


def apply_threshold(scored_pairs, threshold):
    """Filter pairs by threshold.
    
    Args:
        scored_pairs: list of (s1_id, s23_id, score) tuples
        threshold: minimum score to keep
        
    Returns:
        list of (s1_id, s23_id, score) tuples above threshold
    """
    return [(s1, s23, score) for s1, s23, score in scored_pairs if score >= threshold]


def one_to_one_competition(scored_pairs, margin=0.05):
    """Resolve conflicts where an S2/S3 record is matched by multiple S1 entities.
    
    Since S1 is deduplicated, each S2/S3 record should belong to at most one S1.
    Keep only the highest-scoring S1 for each S2/S3 record.
    
    Args:
        scored_pairs: list of (s1_id, s23_id, score) tuples (already above threshold)
        margin: minimum margin over runner-up to keep the assignment.
                If margin is not met, drop from both to protect precision.
    
    Returns:
        list of (s1_id, s23_id, score) tuples after competition
    """
    # Group by s23_id
    s23_to_s1 = defaultdict(list)
    for s1_id, s23_id, score in scored_pairs:
        s23_to_s1[s23_id].append((s1_id, score))
    
    result = []
    for s23_id, assignments in s23_to_s1.items():
        if len(assignments) == 1:
            s1_id, score = assignments[0]
            result.append((s1_id, s23_id, score))
        else:
            # Multiple S1 claim this S2/S3 - competition
            assignments.sort(key=lambda x: -x[1])  # descending by score
            best_s1, best_score = assignments[0]
            runner_up_score = assignments[1][1]
            
            if best_score - runner_up_score >= margin:
                result.append((best_s1, s23_id, best_score))
            # else: drop from all to protect precision
    
    return result


def make_predictions(scored_pairs, all_s1_ids, threshold_first=0.5, 
                     threshold_extra=0.4, competition_margin=0.05):
    """Make final predictions from scored candidate pairs.
    
    Args:
        scored_pairs: list of (s1_id, s23_id, score) tuples
        all_s1_ids: set of all S1 entity IDs (to ensure complete output)
        threshold_first: threshold for first match (stricter)
        threshold_extra: threshold for additional matches (can be looser)
        competition_margin: margin for one-to-one competition
    
    Returns:
        dict {s1_id: set of matched s23_ids}
    """
    # Apply base threshold (use the looser one)
    base_threshold = min(threshold_first, threshold_extra)
    filtered = apply_threshold(scored_pairs, base_threshold)
    
    # Run one-to-one competition
    competed = one_to_one_competition(filtered, margin=competition_margin)
    
    # Group by S1
    s1_matches = defaultdict(list)
    for s1_id, s23_id, score in competed:
        s1_matches[s1_id].append((s23_id, score))
    
    # Apply per-S1 thresholds
    predictions = {}
    for s1_id in all_s1_ids:
        matches = s1_matches.get(s1_id, [])
        if not matches:
            predictions[s1_id] = set()
            continue
        
        # Sort by score descending
        matches.sort(key=lambda x: -x[1])
        
        # First match must clear threshold_first
        if matches[0][1] < threshold_first:
            predictions[s1_id] = set()
            continue
        
        # Accept first match
        accepted = {matches[0][0]}
        
        # Additional matches must clear threshold_extra
        for s23_id, score in matches[1:]:
            if score >= threshold_extra:
                accepted.add(s23_id)
        
        predictions[s1_id] = accepted
    
    return predictions


def tune_thresholds(scored_pairs, ground_truth, all_s1_ids, metric_fn,
                    threshold_grid=None, margin_grid=None):
    """Grid search for best thresholds maximizing macro-F0.5.
    
    Args:
        scored_pairs: list of (s1_id, s23_id, score) tuples
        ground_truth: dict {s1_id: set of true matched IDs}
        all_s1_ids: set of all S1 entity IDs
        metric_fn: function(predictions, ground_truth) -> float
        threshold_grid: list of threshold values to try
        margin_grid: list of margin values to try
    
    Returns:
        dict with best params and score
    """
    if threshold_grid is None:
        threshold_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    if margin_grid is None:
        margin_grid = [0.0, 0.02, 0.05, 0.1]
    
    best_score = -1
    best_params = {}
    
    for t_first in threshold_grid:
        for t_extra in [t for t in threshold_grid if t <= t_first]:
            for margin in margin_grid:
                preds = make_predictions(
                    scored_pairs, all_s1_ids,
                    threshold_first=t_first,
                    threshold_extra=t_extra,
                    competition_margin=margin,
                )
                score = metric_fn(preds, ground_truth)
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'threshold_first': t_first,
                        'threshold_extra': t_extra,
                        'competition_margin': margin,
                        'score': score,
                    }
    
    return best_params


if __name__ == "__main__":
    # Smoke test
    scored = [
        ('S1-001', 'S2-001', 0.9),
        ('S1-001', 'S2-002', 0.7),
        ('S1-002', 'S2-001', 0.6),  # Conflict: S2-001 claimed by S1-001 and S1-002
        ('S1-003', 'S3-001', 0.3),  # Below threshold
    ]
    
    all_s1 = {'S1-001', 'S1-002', 'S1-003', 'S1-004'}
    
    preds = make_predictions(scored, all_s1, 
                             threshold_first=0.5, threshold_extra=0.4,
                             competition_margin=0.05)
    
    assert 'S2-001' in preds['S1-001'], "S2-001 should be matched to S1-001"
    assert 'S2-002' in preds['S1-001'], "S2-002 should also be matched to S1-001"
    assert preds['S1-002'] == set(), "S1-002 lost competition for S2-001"
    assert preds['S1-003'] == set(), "S1-003 below threshold"
    assert preds['S1-004'] == set(), "S1-004 is singleton"
    
    print("Decision layer smoke test PASSED!")
