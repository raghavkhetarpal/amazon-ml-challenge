#!/usr/bin/env python3
"""
Stage 5: F0.5 metric implementation.

Per S1 entity:
    P = |pred ∩ true| / |pred|
    R = |pred ∩ true| / |true|
    F0.5 = 1.25 * P * R / (0.25 * P + R)

Conventions:
    - true empty, pred empty -> 1.0 (correct singleton)
    - true empty, pred non-empty -> 0.0 (false positive on singleton)
    - true non-empty, pred empty -> 0.0 (missed all matches)
    - P + R == 0 -> 0.0

Macro-average over all S1 entities.
"""


def f05_per_entity(pred_set, true_set):
    """Compute F0.5 for a single S1 entity.
    
    Args:
        pred_set: set of predicted matched IDs
        true_set: set of true matched IDs
    
    Returns:
        float: F0.5 score for this entity
    """
    pred_set = set(pred_set) if not isinstance(pred_set, set) else pred_set
    true_set = set(true_set) if not isinstance(true_set, set) else true_set
    
    # Both empty = correct singleton
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    
    # True empty but predicted something = false positive on singleton
    if len(true_set) == 0 and len(pred_set) > 0:
        return 0.0
    
    # True non-empty but predicted nothing = missed all
    if len(true_set) > 0 and len(pred_set) == 0:
        return 0.0
    
    # Both non-empty
    intersection = pred_set & true_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(true_set)
    
    if precision + recall == 0:
        return 0.0
    
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


def macro_f05(predictions, ground_truth):
    """Compute macro-averaged F0.5 over all S1 entities.
    
    Args:
        predictions: dict {s1_id: set of predicted matched IDs}
        ground_truth: dict {s1_id: set of true matched IDs}
    
    Returns:
        float: macro-averaged F0.5
    """
    scores = []
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        true = ground_truth[s1_id]
        scores.append(f05_per_entity(pred, true))
    
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def macro_f05_detailed(predictions, ground_truth, s1_countries=None):
    """Compute macro-averaged F0.5 with per-country and singleton breakdowns.
    
    Args:
        predictions: dict {s1_id: set of predicted matched IDs}
        ground_truth: dict {s1_id: set of true matched IDs}
        s1_countries: optional dict {s1_id: country_label}
    
    Returns:
        dict with 'overall', 'per_country', 'singleton', 'non_singleton' scores
    """
    scores = {}
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        true = ground_truth[s1_id]
        scores[s1_id] = f05_per_entity(pred, true)
    
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    
    # Singleton vs non-singleton
    singleton_scores = [scores[s1] for s1, true in ground_truth.items() if len(true) == 0]
    non_singleton_scores = [scores[s1] for s1, true in ground_truth.items() if len(true) > 0]
    
    result = {
        'overall': overall,
        'singleton': sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0,
        'non_singleton': sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0,
        'n_entities': len(scores),
        'n_singleton': len(singleton_scores),
        'n_non_singleton': len(non_singleton_scores),
    }
    
    # Per country
    if s1_countries:
        country_scores = {}
        for s1_id, score in scores.items():
            c = s1_countries.get(s1_id, "UNKNOWN")
            if c not in country_scores:
                country_scores[c] = []
            country_scores[c].append(score)
        
        result['per_country'] = {
            c: sum(s) / len(s) for c, s in country_scores.items()
        }
    
    return result


# ---- Unit tests ----
def test_metric():
    """Test the metric with the worked example from the problem statement."""
    # Example: pred {S2-00047, S2-00193, S3-00812} vs truth {S2-00047, S3-00812}
    # P = 2/3, R = 2/2 = 1.0, F0.5 = 1.25 * (2/3) * 1.0 / (0.25 * (2/3) + 1.0) = 0.714...
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    score = f05_per_entity(pred, true)
    assert abs(score - 0.714) < 0.001, f"Expected ~0.714, got {score}"
    
    # Singleton correct: both empty -> 1.0
    assert f05_per_entity(set(), set()) == 1.0
    
    # Singleton incorrect: pred non-empty, true empty -> 0.0
    assert f05_per_entity({"S2-001"}, set()) == 0.0
    
    # Missed all: pred empty, true non-empty -> 0.0
    assert f05_per_entity(set(), {"S2-001"}) == 0.0
    
    # Perfect match
    assert f05_per_entity({"S2-001"}, {"S2-001"}) == 1.0
    
    # Macro average test
    gt = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": set(),  # singleton
        "S1-003": {"S2-003"},
    }
    preds = {
        "S1-001": {"S2-001", "S3-001"},  # perfect -> 1.0
        "S1-002": set(),                  # correct singleton -> 1.0
        "S1-003": set(),                  # missed -> 0.0
    }
    macro = macro_f05(preds, gt)
    expected = (1.0 + 1.0 + 0.0) / 3
    assert abs(macro - expected) < 1e-6, f"Expected {expected}, got {macro}"
    
    print("All metric tests PASSED!")


if __name__ == "__main__":
    test_metric()
