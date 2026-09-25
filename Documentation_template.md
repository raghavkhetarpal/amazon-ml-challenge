# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM]  
**Team Members:** Antigravity Pair Programming Team  
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary
We designed and implemented a production-grade, highly scalable multi-channel entity resolution pipeline capable of resolving 1.73 million test entities across 11.7 million noisy multi-source records under tight memory (<3 GB RAM) and offline constraints. Our architecture combines multi-channel lexical and phonetic blocking (achieving **99.13% candidate recall** in seconds via CSR sparse vectorization), a 42-feature pairwise gradient boosting model (LightGBM) trained on hard negatives with GroupKFold cross-validation (**validation macro-F0.5 = 0.9670**; US = 0.9839, India = 0.9416), and a post-scoring bipartite competition layer enforcing deduplication with strict confidence margin gating. To ensure zero-shot cross-border generalization to the unseen **France** test partition, our representations rely exclusively on language-agnostic token geometries, Unicode NFKD decomposition, and generic hierarchical regexes, demonstrated by a minimal **0.0382** Leave-One-Country-Out transfer gap.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12,527,040 training records and 11,702,133 test records revealed five fundamental structural insights:
1. **Deduplication Invariant:** Ground truth audits across 2,206,821 Source 1 entities confirmed that **0** Source 2 or Source 3 records belong to more than one Source 1 entity. S1 acts as a strict reference authority; resolving cross-entity conflicts via 1-to-1 competitive assignment eliminates false merges.
2. **Strict Geographic Partitioning:** Ground-truth matches across 7,638,365 pairs showed exactly **0 cross-country matches**. Business entity resolution can be factored strictly by country partition, enabling memory footprint reduction from >18 GB to <3 GB via independent country streaming.
3. **Severe Cross-Script Asymmetry:** In India, Source 2 records frequently use Devanagari, Tamil, or Telugu transliterated scripts for legal entity names, whereas Source 1 and Source 3 use Latin script. Name-only character n-gram blocking suffers recall collapse (89.33%). However, addresses consistently share alphanumeric tokens, building numbers, and municipal landmarks in English, enabling multi-channel address fusion to restore candidate recall to **99.13%**.
4. **Macro-F0.5 Metric Properties:** The evaluation metric heavily penalizes false positives ($\beta = 0.5$ weights precision 2x over recall). Furthermore, correctly identifying singletons yields 1.0, while any predicted match on a true singleton degrades its score to 0.0. The optimal operating threshold on model probability is conservative ($T \ge 0.85 - 0.90$).
5. **Open-Set Country Shift (France):** Unseen in training, France introduces diacritics (é, è, ê, ç), specific legal forms (`SARL`, `SAS`, `SA`, `EURL`, `SCI`), and 5-digit postal codes (`75001`). The pipeline must never rely on country categorical features or one-hot vectors.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel Blocking + Pairwise Gradient Boosted Trees (LightGBM) + Bipartite One-to-One Competition + Dual-Threshold Calibration.  
**Core Innovation:** Direct CSR array indexing (`indptr` slicing) for 30x faster sparse matrix top-$k$ candidate generation, combined with a bipartite competition resolution layer that strictly enforces reference deduplication and filters ambiguous runner-up matches.

```
+-----------------------------------------------------------------------------------+
|                           INPUT SOURCE RECORDS (S1, S2, S3)                      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 1: Language-Agnostic Normalization (Unicode NFKD, Legal/Address Expansions) |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 2: Multi-Channel Candidate Blocking (Word TF-IDF + Addr TF-IDF + Hash Keys) |
|         -> 99.13% Pairwise Recall, 60 candidates/S1, 0.999+ Reduction Ratio        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 3: Pairwise Feature Engine (42 Name, Address, Phonetic & Contextual Metrics) |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 4: LightGBM Gradient Boosting Classifier (Trained on In-Distribution Negs)  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| STAGE 5: Bipartite Competition & Thresholding (1-to-1 Greedy Margin, T_first=0.90)|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| OUTPUT: matching_results.tsv (Precision F0.5) & candidate_pairs.tsv (Valid Subset)|
+-----------------------------------------------------------------------------------+
```

---

## 3. Candidate Generation (Blocking)

To reduce the 17.2-trillion naive pairwise comparison space to a scalable sub-linear candidate set, we constructed a high-recall multi-channel blocking engine:

- **Blocking channels used:**
  1. **Name Word TF-IDF (1-2 grams):** Sublinear term frequency weighting with sparse cosine similarity top-$k$ retrieval via optimized CSR `indptr` slicing.
  2. **Address Word TF-IDF (1-2 grams):** Recovers cross-script matches in multilingual environments where business names differ in alphabet but street/locality tokens coincide.
  3. **First-Token Inverted Index:** Inverted hash index on normalized leading entity tokens (minimum 3 characters, stop-tokens pruned).
  4. **Phonetic Encoding Index:** Simplified double-metaphone consonant skeleton index invariant to vocalic and phonetic transliteration shifts.
  5. **Postal Code Direct Match:** Exact hash join on extracted 5-6 digit postal codes (IN PIN, US ZIP, FR code).
- **Candidate pairs generated:**
  - Evaluated on validation: ~60 candidates per S1 entity (reduction ratio: **0.9989**).
  - Test set candidate generation: capped at top-25 highest-confidence candidates per S1 entity (~43 million candidate pairs across 1.73M S1 queries).
- **How true matches were preserved:**
  - Independent channel union guarantees that if a pair fails the name channel due to heavy spelling corruption or transliteration, the address channel or phonetic/postal hash joins recover it.
  - Candidate recall reached **99.13%** on validation benchmarks (6,864 of 6,924 true pairs captured).

---

## 4. Matching Model

### Features Used (42 Pairwise Metrics)
1. **Name Similarity Features (16):**
   - Token Jaccard, Sorted-Token Jaccard (transposition-invariant)
   - Character 3-gram Jaccard, Token Sort Ratio, Token Set Ratio (`rapidfuzz`)
   - Levenshtein Normalized Similarity, Jaro-Winkler Distance
   - First-token equality, Phonetic-key equality
   - Token containment ratio (shorter string subset in longer string)
   - Length ratio, Token count ratio and absolute difference
   - Legal form equality, Legal form conflict flag, Both-empty legal flag
2. **Address Similarity Features (16):**
   - Address token Jaccard, Address character 3-gram Jaccard
   - Address fuzzy ratio, partial ratio, token sort ratio
   - Postal code exact match, 3-digit prefix match, postal conflict flag
   - House/building number exact match, house number conflict flag
   - Address missingness indicators (S1 empty, S23 empty, both empty)
   - Address length ratio
3. **Contextual & Structural Features (10):**
   - Blocking cosine score
   - Candidate rank within query candidate list
   - Score margin between top-1 and top-2 candidate
   - Total candidates generated for S1
   - Source indicator flags (`is_S2`, `is_S3`)
   - Empty name flags

### Model Architecture & Training
- **Model Type:** LightGBM Gradient Boosted Decision Trees (GBDT), `objective: binary`, `metric: binary_logloss`.
- **Hyperparameters:** `num_leaves: 63`, `learning_rate: 0.1`, `feature_fraction: 0.8`, `bagging_fraction: 0.8`, `min_child_samples: 50-100`, `lambda_l1: 0.1`, `lambda_l2: 0.1`.
- **Negative Sampling:** The model was trained directly on the hard negatives produced by our candidate generation blocking engine, matching the exact inference distribution.
- **Validation Splitting:** 3-fold `GroupKFold` grouped strictly by `source1_entity_id` to prevent data leakage between candidate pairs of the same entity.

### Threshold Selection Method
- Due to the F0.5 precision-heavy weighting and the 1.0 payoff for true singletons, optimal thresholds were chosen by grid search maximizing out-of-fold macro-F0.5 directly.
- **Calibrated Decision Layer:**
  - $T_{\text{first}} = 0.90$: High-confidence threshold required to admit the first match for an S1 entity (safeguarding singletons).
  - $T_{\text{extra}} = 0.90$: Confirmatory threshold for subsequent matches.
  - $\text{Margin} = 0.05$: In one-to-one competition, an S2/S3 entity is awarded to an S1 only if its score exceeds the runner-up S1 by at least 0.05.

---

## 5. Results & Error Analysis

### Performance Metrics

| Metric / Evaluation Split | Score |
|:---|:---:|
| **Overall Macro-F0.5 (Validation)** | **0.9670** |
| Non-Singleton Macro-F0.5 | **0.9695** |
| Singleton Macro-F0.5 | **0.9266** |
| US Partition Macro-F0.5 | **0.9839** |
| India Partition Macro-F0.5 | **0.9416** |
| **Blocking Recall Ceiling** | **99.13%** |
| **Blocking Reduction Ratio** | **0.9989** |

### Leave-One-Country-Out (LOCO) Generalization
To simulate deployment on the unseen **France** test partition:
- **Trained on India $\rightarrow$ Evaluated on US:** Macro-F0.5 = **0.9792** (In-domain: 0.9985, Gap: +0.0193)
- **Trained on US $\rightarrow$ Evaluated on India:** Macro-F0.5 = **0.9158** (In-domain: 0.9730, Gap: +0.0571)
- **Mean Cross-Country Transfer Gap:** **0.0382** (demonstrates robust cross-border feature invariance).

### Error Analysis
- **False Positives (Wrong Merges):**
  - Chain businesses and retail franchises sharing identical brand names in adjacent municipal sectors with missing or generic unit numbers (e.g., "Subway" or "State Bank ATM"). Mitigated by house number conflict penalties and postal code prefix checking.
- **False Negatives (Missed Matches):**
  - Records where Source 3 provided only a bare internet domain URL (e.g. `desai-alanah.com`) with no street address, causing both name token ratios and address Jaccard to fail. Handled where possible by character n-gram partial ratios and postal code indexing.

---

## 6. France Generalization Strategy
No training labels exist for France. We enforced the following zero-shot principles:
1. **Diacritic & Unicode Normalization:** Unicode NFKD decomposition strips accents (`Société` $\rightarrow$ `societe`, `HÊTRES` $\rightarrow$ `hetres`).
2. **French Legal Suffix Canonicalization:** Mapped `SARL`, `SAS`, `SA`, `EURL`, `SCI`, `SCP`, `SNC`, `SASU` to canonical tokens.
3. **French Address Lexicon:** Normalized abbreviations including `rue`, `av`/`ave` (avenue), `bd`/`bld` (boulevard), `all` (allée), `imp` (impasse), `che` (chemin), and `cedex`.
4. **Hierarchical 5-Digit Postal Codes:** Generic regex extracts 5-digit French postal codes (`75001`), supporting exact and 2-digit departmental prefix alignment.
5. **No Categorical Country Features:** Zero country one-hot features in the model, ensuring identical inference pathways for France as for US/India.

---

## 7. Model Licenses & Constraints Compliance

| Model / Library | Version | License | Parameters / Size | Compliance Note |
|:---|:---:|:---:|:---:|:---|
| **LightGBM** | 4.6.0 | MIT License | GBDT (Tree-based) | Permissive license, non-neural |
| **scikit-learn** | 1.6.1 | BSD-3-Clause | Classical ML | Permissive, CPU vectorized |
| **rapidfuzz** | 3.13.0 | MIT License | C++ String Metrics | Permissive, high performance |
| **scipy / numpy** | 1.13.1 / 2.0.2 | BSD-3-Clause | Scientific Stack | Standard open source |
| **pandas** | 2.3.3 | BSD-3-Clause | Data Processing | Standard open source |

*Zero external lookups, geocoding APIs, or web registries were used. The pipeline is 100% offline self-contained.*

---

## 8. Conclusion
We delivered an end-to-end, reproducible solution for large-scale business entity resolution. By optimizing sparse CSR candidate retrieval (yielding 99.13% blocking recall in seconds), extracting 42 structural string and address features, training an in-distribution LightGBM classifier, and enforcing bipartite one-to-one reference deduplication, we achieved **0.9670 validation macro-F0.5** with exceptional cross-country stability.

---

## Appendix

### A. Code Artefacts & Structure
The submission package layout:
```
code/business_entity_resolution/
├── README.md                      # End-to-end reproduction guide
├── requirements.txt               # Pinned Python dependencies
└── src/
    ├── __init__.py                # Package root
    ├── normalize.py               # Unicode, legal suffix, address regexes
    ├── blocking.py                # Multi-channel CSR top-k blocking engine
    ├── features.py                # 42 pairwise similarity features
    ├── decide.py                  # One-to-one bipartite competition
    ├── metric.py                  # Macro-F0.5 competition metric
    ├── eda.py                     # Stage 0 exploratory analysis
    ├── evaluate_loco.py           # Leave-One-Country-Out benchmark
    ├── predict_test.py            # Streaming test prediction engine
    └── run_pipeline.py            # Unified CLI entrypoint
```

### B. Reproduction Commands
1. **Smoke Test:**
   ```bash
   python3 code/business_entity_resolution/src/run_pipeline.py --mode predict --sample 500
   ```
2. **Leave-One-Country-Out (LOCO) Evaluation:**
   ```bash
   python3 code/business_entity_resolution/src/evaluate_loco.py --sample 3000
   ```
3. **Full Test Prediction & Validation:**
   ```bash
   python3 code/business_entity_resolution/src/predict_test.py --batch-size 25000
   python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```
