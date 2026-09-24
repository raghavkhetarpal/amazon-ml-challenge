# Notes & Assumptions

## Data Paths
- Training data: `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`
- Test data: `dataset/test/test_source{1,2,3}.tsv`
- Output: `output/matching_results.tsv`, `output/candidate_pairs.tsv`

## Key EDA Findings
- Train: S1=2,206,821, S2=5,034,616, S3=5,285,603
- Test: S1=1,732,544, S2=4,887,273, S3=5,082,316
- Singletons: 5.6% (123,247 of 2,206,821)
- Mean matches per S1: 3.46, median: 3.0, max: 11
- **Deduplication confirmed:** 0 S2/S3 IDs assigned to >1 S1
- **No cross-country matches:** all matches are within the same country
- Country labels: Train={US, India}, Test={US, India, France}
- S2 has Devanagari/Tamil/Telugu script names (transliterations of English names)
- S3 has domain names (e.g., `maurewilliamscolombier.com`), garbled text, word transpositions
- Missing addresses: S2 ~3.4%, S3 ~3.3%

## Assumptions Made
1. Blocking within country is safe (no cross-country matches in training data)
2. One-to-one assignment is valid (deduplication confirmed)
3. Fitting TF-IDF on union of train+test text is acceptable (unsupervised, no labels)
4. French data patterns will be similar to US/India in terms of noise types
5. S2 Hindi/Tamil names won't match French entities (handled by country blocking)

## Pretrained Models Used
| Model | License | Parameters | Purpose |
|-------|---------|------------|---------|
| None (classical pipeline only) | N/A | N/A | Using TF-IDF + LightGBM only |

LightGBM: MIT license, classical gradient boosting (not a pretrained language model).

## France Generalization Strategy
- All normalization is language-agnostic (Unicode NFKD + accent stripping)
- French legal suffixes added: SARL, SAS, SA, EURL, SCI, SCP, SNC, SASU
- French address abbreviations: rue, av/avenue, bd/bld/boulevard, imp/impasse, cedex
- 5-digit French postal codes handled by generic postal code regex
- No country-specific features in the model (no country one-hot)
- TF-IDF char n-grams work on any script/language
