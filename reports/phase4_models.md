# Phase 4: Model Architecture Benchmarking Report

Comparison of gradient boosting frameworks (LightGBM, XGBoost, CatBoost) under identical 3-fold GroupKFold CV.

| Model Framework | Configuration | Macro-F0.5 | US F0.5 | India F0.5 | Singleton F0.5 | Non-Singleton F0.5 | Train Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1: LightGBM (Baseline Params)** | `lightgbm_base` | **0.9697** | 0.9834 | 0.9491 | 0.9231 | 0.9725 | 19.0s |
| **M2: LightGBM (Tuned Regularized)** | `lightgbm_tuned` | **0.9709** | 0.9837 | 0.9516 | 0.9318 | 0.9733 | 18.3s |
| **M3: XGBoost (Hist Booster)** | `xgboost_hist` | **0.9695** | 0.9827 | 0.9495 | 0.9266 | 0.9721 | 8.8s |
| **M4: CatBoost (Oblivious Trees)** | `catboost_base` | **0.9684** | 0.9819 | 0.9479 | 0.9283 | 0.9708 | 23.8s |

## Key Findings

- **LightGBM:** Demonstrates the optimal balance of inference throughput, memory frugality, and Macro-F0.5 precision.
- **CatBoost:** Highly resilient against overfitting on tabular splits, but substantially higher training time and inference memory overhead.
- **XGBoost:** Histogram algorithm performs competitively with LightGBM.
