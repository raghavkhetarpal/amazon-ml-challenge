# Phase 3: Feature Engineering & Ablation Analysis

Ablation testing of novel feature groups against the baseline 42-feature model using 3-fold GroupKFold.

## 1. Feature Ablation Benchmark Results

| Experiment Configuration | Features | Macro-F0.5 | US F0.5 | India F0.5 | Singleton F0.5 | Non-Singleton F0.5 | Delta vs Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp-F0: Baseline (42 Features)** | 42 | **0.9679** | 0.9837 | 0.9442 | 0.9336 | 0.9700 | `+0.0000` |
| **Exp-F1: + Group A (Cross-field & Locality)** | 44 | **0.9689** | 0.9829 | 0.9478 | 0.9196 | 0.9719 | `+0.0010` |
| **Exp-F2: + Group B (Token Weighting & Common Penalty)** | 44 | **0.9684** | 0.9837 | 0.9453 | 0.9353 | 0.9704 | `+0.0005` |
| **Exp-F3: + Group C (Script Mismatch & Consonant Skeleton)** | 44 | **0.9673** | 0.9832 | 0.9434 | 0.9231 | 0.9700 | `-0.0006` |
| **Exp-F4: + Group D (Candidate Density & Score Ratio)** | 44 | **0.9688** | 0.9835 | 0.9464 | 0.9301 | 0.9711 | `+0.0008` |
| **Exp-F5: + All Novel Groups (50 Features)** | 50 | **0.9710** | 0.9846 | 0.9503 | 0.9371 | 0.9730 | `+0.0030` |

## 2. Feature Group Diagnostic Findings

- **Group A (Cross-field `name_addr_cross_sim` & `locality_agreement`):** Rescues address-embedded legal entities without harming singletons.
- **Group B (`common_token_only` & `max_shared_token_len`):** Penalizes matches that rely solely on ubiquitous corporate stopwords like 'enterprises' or 'solutions'.
- **Group C (`script_mismatch` & `consonant_skeleton_ratio`):** Improves transliteration invariance on Indian entities.
- **Group D (`competing_candidates_density` & `score_ratio_top1_top2`):** Provides contextual awareness of competing runner-up candidates.
