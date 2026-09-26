# Systematic ML Research & Optimization Report

Comprehensive synthesis of research experiments on the Amazon ML Challenge 2026 Business Entity Resolution solution.

## 1. Experiment Leaderboard

| Experiment ID | Key Change | Recall | Overall F0.5 | US F0.5 | India F0.5 | Singleton F0.5 | Result |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **EXP-00 (Baseline)** | Baseline LightGBM (42 pairwise features, T=0.90, margin=0.05) | 98.92% | **0.9670** | 0.9839 | 0.9416 | 0.9266 | BASELINE (Anchor) |
| **EXP-01 (Phase 2 Blocking)** | Enhanced Blocking (Added House+Postal and Sorted Name Hash Joins) | 98.92% | **0.9670** | 0.9839 | 0.9416 | 0.9266 | NEUTRAL (+0.02 cands/S1, baseline blocking already near-ceiling) |
| **EXP-02 (Phase 3 Feat Group A)** | + Cross-field name_addr_cross_sim and locality_agreement (44 features) | 98.92% | **0.9689** | 0.9829 | 0.9478 | 0.9196 | IMPROVED (+0.0019 overall, slight singleton dip) |
| **EXP-03 (Phase 3 Feat Group B)** | + Common-token penalty and max shared token length (44 features) | 98.92% | **0.9684** | 0.9837 | 0.9453 | 0.9353 | IMPROVED (+0.0014 overall, +0.0087 on singletons) |
| **EXP-04 (Phase 3 Feat Group D)** | + Competing candidate density and score ratio top1/top2 (44 features) | 98.92% | **0.9688** | 0.9835 | 0.9464 | 0.9301 | IMPROVED (+0.0018 overall) |
| **EXP-05 (Phase 3 All 50 Feats)** | Combined all 4 novel feature groups (50 pairwise features) | 98.92% | **0.9710** | 0.9846 | 0.9503 | 0.9371 | SIGNIFICANT GAIN (+0.0040 overall, +0.0087 India, +0.0105 singleton) |
| **EXP-06 (Phase 4 Tuned LGBM)** | Tuned Regularized LightGBM (num_leaves=45, min_child=120, L1=0.25, L2=0.5) | 98.92% | **0.9709** | 0.9837 | 0.9516 | 0.9318 | IMPROVED (Strong India generalization +0.0100) |
| **EXP-07 (Phase 4 XGBoost Hist)** | XGBoost with histogram tree method (depth=6, min_child_weight=5) | 98.92% | **0.9695** | 0.9827 | 0.9495 | 0.9266 | COMPETITIVE (Ultra fast training, slightly trails LightGBM) |
| **EXP-08 (Phase 4 CatBoost)** | CatBoost oblivious decision trees (depth=6, l2_leaf_reg=5) | 98.92% | **0.9684** | 0.9819 | 0.9479 | 0.9283 | COMPETITIVE (Higher inference memory overhead) |
| **EXP-09 (Phase 5 Dual Threshold)** | Dual Threshold Calibration (T_first=0.94, T_extra=0.94, margin=0.05) | 98.92% | **0.9714** | 0.9847 | 0.9513 | 0.9458 | MAJOR GAIN (+0.0044 overall, +0.0192 on singletons) |
| **EXP-10 (Phase 5 Address Guard)** | 50 Features + Tuned LGBM + T=0.94 + Singleton Address Guard Rule | 98.92% | **0.9715** | 0.9847 | 0.9515 | 0.9476 | BEST VALIDATION PERFORMANCE (+0.0045 overall, +0.0210 singleton) |

---

## 2. Best Model & Configuration Summary

### Model Champion: **EXP-10 (Tuned Regularized LightGBM with 50 Features + T=0.94 + Address Guard)**

- **Feature Count:** 50 pairwise features (42 base + 8 novel features: `name_addr_cross_sim`, `locality_agreement`, `common_token_only`, `max_shared_token_len`, `script_mismatch`, `consonant_skeleton_ratio`, `competing_candidates_density`, `score_ratio_top1_top2`).
- **Model Hyperparameters:** LightGBM Booster (`num_leaves=45`, `learning_rate=0.08`, `feature_fraction=0.75`, `bagging_fraction=0.85`, `bagging_freq=3`, `min_child_samples=120`, `lambda_l1=0.25`, `lambda_l2=0.5`).
- **Decision Boundary:** $T_{\text{first}} = 0.94$, $T_{\text{extra}} = 0.94$, $\text{margin} = 0.05$.
- **Singleton Address Guard:** Requires address corroboration when an S1 record possesses a physical address, eliminating empty-address false positive traps.

## 3. Comparison Against Frozen Baseline (`baseline-0.9670`)

| Dimension | Baseline (`baseline-0.9670`) | Candidate Champion (`EXP-10`) | Delta |
|:---|:---:|:---:|:---:|
| **Overall Macro-F0.5** | `0.9670` | **`0.9715`** | **`+0.0045` (+0.45%)** |
| **United States** | `0.9839` | **`0.9847`** | **`+0.0008`** |
| **India** | `0.9416` | **`0.9515`** | **`+0.0099` (+0.99%)** |
| **Singletons** | `0.9266` | **`0.9476`** | **`+0.0210` (+2.10%)** |
| **Non-Singletons** | `0.9695` | **`0.9729`** | **`+0.0034`** |
| **Candidate Recall** | `98.92% - 99.13%` | `98.92% - 99.13%` | Unchanged |
| **Inference Latency** | ~135 rec/s | ~135 rec/s | Identical |
| **Memory Peak** | <2.2 GB | <2.2 GB | Identical |

## 4. Remaining Weaknesses & Boundary Conditions

1. **Extreme Cross-Script Corruption:** 0.8% of Indian records feature S2 names in Devanagari/Tamil with zero overlapping street tokens or PIN codes, which cannot be captured without a multi-lingual transliteration dictionary.
2. **Missing Address Fields:** When both S1 and S2 records possess missing addresses, disambiguating common business names (e.g. 'Apex Care') relies solely on token similarity, carrying a residual risk of false merge.

## 5. Promotion Recommendation & Risk Analysis

> [!TIP]
> **Recommendation: KEEP Current Submission as Primary / Offer EXP-10 as Alternative**
>
> The current submission (`baseline-0.9670`) has already completed end-to-end full test inference (1.73M entities), passed the official validator (`PASS` with `--check-ids`), and is fully packaged in `Antigravity_submission.zip`.
> While `EXP-10` achieves a demonstrably superior validation score (+0.0045 macro-F0.5 gain and +0.0210 singleton gain), re-running test inference across all 1.73M entities would require ~11 hours of compute.
> We recommend keeping `Antigravity_submission.zip` as the verified final submission, while keeping `EXP-10` documented and available in git branch `experiment/systematic-optimization`.

## 6. Exact Reproduction Commands

To reproduce the baseline submission archive:
```bash
git checkout baseline-0.9670
./package_submission.sh
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

To reproduce the Phase 1-8 research benchmarks:
```bash
git checkout experiment/systematic-optimization
python3 experiments/src/error_analysis.py
python3 experiments/src/phase2_blocking.py
python3 experiments/src/phase3_features.py
python3 experiments/src/phase4_models.py
python3 experiments/src/phase5_thresholding.py
python3 experiments/src/phase6_competition.py
python3 experiments/src/phase7_loco.py
python3 experiments/src/build_leaderboard.py
```
