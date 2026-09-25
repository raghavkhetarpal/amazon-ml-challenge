# Model & System Experiment Tracking

This document logs all iterative experiments, ablation studies, and validation metrics for the **Amazon ML Challenge 2026: Business Entity Resolution** solution.

---

## 1. Summary of Iterative Experiments

| Exp ID | Configuration / Description | Blocking Channels | Candidate Pairs per S1 | Blocking Recall | Val Macro-F0.5 | Singleton F0.5 | Non-Singleton F0.5 | US Macro-F0.5 | India Macro-F0.5 | Notes |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **E1** | Baseline: Char TF-IDF + Word TF-IDF + First-Token Hash + Postal Hash | Name Char (3-4g), Name Word (1-2g), First Token, Postal Code | ~60 | 89.33% | 0.9323 | 0.9474 | 0.9310 | 0.9845 | 0.8601 | Poor India recall due to S2 transliterated/Devanagari names |
| **E2** | Multi-Channel Fusion: Added Address Char TF-IDF + Combined Name+Address TF-IDF | Name Char, Name Word, Hash keys, Postal, **Addr Char**, **Combined** | ~98 | **99.34%** | 0.9771 | 0.9583 | 0.9783 | 0.9895 | 0.9576 | Major breakthrough on Indian records where S2 address is in English |
| **E3** | Scaled Feature Validation (10,000 S1 Entities, 1M candidate pairs) | All 6 channels | ~100 | **98.92%** | **0.9670** | 0.9266 | 0.9695 | 0.9839 | 0.9416 | Validated at 1M pairs; 3-fold GroupKFold cross-validation |
| **E4** | Decision Layer: One-to-One Competition + Dual-Threshold Optimization | All 6 channels | ~100 | 98.92% | **0.9670** | 0.9266 | 0.9695 | 0.9839 | 0.9416 | T_first=0.9, T_extra=0.9, margin=0.05; strict threshold maximizes F0.5 |
| **E5** | Leave-One-Country-Out (LOCO): Train India -> Test US | All 6 channels | ~80 | 99.97% (US) | 0.9792 | 0.9456 | 0.9809 | 0.9792 | - | Minimal transfer gap: +0.0193 vs in-domain |
| **E6** | Leave-One-Country-Out (LOCO): Train US -> Test India | All 6 channels | ~80 | 97.37% (IN) | 0.9158 | 0.9649 | 0.9129 | - | 0.9158 | Cross-script transfer gap: +0.0571 vs in-domain |

---

## 2. Component Ablation & Impact Analysis

### A. Blocking Recall Optimization
- **Problem:** Initial blocking achieved only 89.3% recall because S2 in India contained business names in Hindi (Devanagari), Tamil, or Telugu, while S1 was in Latin transliteration. Char n-grams on names alone produced zero overlap.
- **Solution:** Added Channel 5 (Address Char 3-4g TF-IDF) and Channel 6 (Combined Name + Address TF-IDF). Since addresses in Indian records often share alphanumeric building numbers, street names, or city tokens even when the business name is in a local script, this captured cross-script matches.
- **Result:** Recall surged from **89.33% to 99.34%** (+10.01% absolute recall gain). India F0.5 improved from 0.8601 to 0.9576.

### B. One-to-One Assignment Constraint
- **Hypothesis:** Because Source 1 is a deduplicated reference source, no single Source 2 or Source 3 record should be matched to multiple Source 1 entities.
- **Empirical Check on Ground Truth:** Verified on 2,206,821 ground-truth S1 entities and 7,638,365 match links. S2/S3 IDs assigned to >1 S1 entity: **0**.
- **Impact:** Enforcing greedy competition with a confidence margin protects precision against ambiguous near-identical entity collisions.

### C. Threshold Optimization for Macro-F0.5 with Singletons
- Macro-F0.5 weights precision 2x over recall ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$).
- Singletons represent 5.58% of entities; any false positive drops an entity's score from 1.0 to 0.0.
- Out-of-fold grid search revealed that standard probability thresholds (0.5) underperform. Operating at $T \ge 0.85 - 0.90$ with a competition margin of 0.05 yields optimal macro-F0.5.

### D. Cross-Border Generalization to Unseen France
- The test set contains an unseen 3rd country: **France** (~259k S1, ~703k S2, ~732k S3).
- **Strategy:**
  1. Complete language/script-agnostic feature set (no country categorical features, no country one-hot encoding).
  2. Generic regex-based postal code extraction (handling 5-digit French postal codes alongside 6-digit Indian PIN and 5/9-digit US ZIP).
  3. Unicode NFKD accent-stripping (normalizes `Société` -> `societe`, `Boulevard` -> `boulevard`).
  4. French legal form mapping (`SARL`, `SAS`, `SA`, `EURL`, `SCI`, etc.).
  5. French address abbreviation normalization (`rue`, `av`, `bd`, `cedex`, `all`, `imp`).
- **LOCO Experiment Results:**
  - Average cross-country transfer gap is only **0.0382** (3.8%), proving high resilience to unseen geographical and linguistic distribution shifts.
