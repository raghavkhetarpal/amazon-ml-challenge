# Business Entity Resolution Pipeline

## Overview
Multi-channel blocking + LightGBM classifier for matching business records across three data sources.

## Prerequisites
- Python 3.9+
- macOS / Linux
- ~16 GB RAM for full dataset processing

## Setup
```bash
pip install -r requirements.txt
# On macOS, LightGBM needs OpenMP:
brew install libomp
```

## Data Layout
Place the dataset files at:
```
dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## Running

### Quick smoke test (~5 min)
```bash
python src/run_pipeline.py --mode all --sample 1000
```

### Full training + evaluation (~2-4 hours)
```bash
python src/run_pipeline.py --mode train
```

### Full test prediction (~2-4 hours)
```bash
python src/run_pipeline.py --mode predict
```

### End-to-end (train + predict)
```bash
python src/run_pipeline.py --mode all
```

### EDA only
```bash
python src/run_pipeline.py --mode eda
```

## Output Files
- `output/matching_results.tsv` - Final entity matches (scored on leaderboard)
- `output/candidate_pairs.tsv` - Blocking candidate set

## Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline Stages
1. **EDA** (`eda.py`): Data exploration and statistics
2. **Normalization** (`normalize.py`): Unicode, legal suffix, abbreviation handling
3. **Blocking** (`blocking.py`): Multi-channel TF-IDF + hash-based candidate generation
4. **Features** (`features.py`): 40+ pairwise similarity features
5. **Model** (`run_pipeline.py`): LightGBM gradient boosting with GroupKFold CV
6. **Decision** (`decide.py`): Threshold tuning + one-to-one competition
7. **Metric** (`metric.py`): F0.5 macro-averaged per S1 entity

## Models & Licenses
- **LightGBM** (MIT License) - Gradient boosting classifier
- No pretrained neural models are used in the default pipeline

## Expected Runtimes
- Normalization: ~30 min per source file on full data
- Blocking: ~15-30 min per country
- Feature computation: ~30-60 min
- Model training: ~10-20 min
- Prediction: ~2-4 hours total
