#!/usr/bin/env python3
"""Stage 0: Exploratory Data Analysis for Business Entity Resolution."""

import csv
import os
import sys
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TRAIN_DIR = os.path.join(ROOT, "dataset/train")
TEST_DIR = os.path.join(ROOT, "dataset/test")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_tsv(path):
    """Load a TSV file with proper settings."""
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_values=[])


def run_eda():
    print("=" * 80)
    print("STAGE 0: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Load training data
    print("\n[1] Loading training data...")
    s1_train = load_tsv(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    s2_train = load_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"))
    s3_train = load_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"))
    gt = load_tsv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))

    # Load test data
    print("[2] Loading test data...")
    s1_test = load_tsv(os.path.join(TEST_DIR, "test_source1.tsv"))
    s2_test = load_tsv(os.path.join(TEST_DIR, "test_source2.tsv"))
    s3_test = load_tsv(os.path.join(TEST_DIR, "test_source3.tsv"))

    report = []
    report.append("# EDA Report: Business Entity Resolution\n")

    # ---- Row counts ----
    report.append("## 1. Dataset Scale\n")
    report.append("### Row Counts\n")
    report.append("| Source | Train | Test |")
    report.append("|--------|-------|------|")
    report.append(f"| S1 | {len(s1_train):,} | {len(s1_test):,} |")
    report.append(f"| S2 | {len(s2_train):,} | {len(s2_test):,} |")
    report.append(f"| S3 | {len(s3_train):,} | {len(s3_test):,} |")
    report.append(f"| Ground Truth | {len(gt):,} | N/A |")
    report.append(f"| **Total** | **{len(s1_train)+len(s2_train)+len(s3_train):,}** | **{len(s1_test)+len(s2_test)+len(s3_test):,}** |")
    report.append("")

    print(f"  Train: S1={len(s1_train):,}, S2={len(s2_train):,}, S3={len(s3_train):,}, GT={len(gt):,}")
    print(f"  Test:  S1={len(s1_test):,}, S2={len(s2_test):,}, S3={len(s3_test):,}")

    # ---- Country distribution ----
    report.append("### Country Distribution\n")
    for name, df in [("Train S1", s1_train), ("Train S2", s2_train), ("Train S3", s3_train),
                     ("Test S1", s1_test), ("Test S2", s2_test), ("Test S3", s3_test)]:
        vc = df["country"].value_counts()
        report.append(f"**{name}:** {dict(vc)}\n")
        print(f"  {name} countries: {dict(vc)}")

    # ---- ID formats ----
    report.append("## 2. ID Formats\n")
    for name, df in [("Train S1", s1_train), ("Train S2", s2_train), ("Train S3", s3_train)]:
        sample = df["entity_id"].head(5).tolist()
        report.append(f"**{name}:** e.g. `{sample[:3]}`")
    report.append("")

    # ---- Nulls / empties ----
    report.append("## 3. Missing / Empty Fields\n")
    report.append("| Source | Field | Empty Count | % |")
    report.append("|--------|-------|-------------|---|")
    for name, df in [("Train S1", s1_train), ("Train S2", s2_train), ("Train S3", s3_train),
                     ("Test S1", s1_test), ("Test S2", s2_test), ("Test S3", s3_test)]:
        for col in ["business_name", "business_address", "country"]:
            empty = (df[col].str.strip() == "").sum()
            pct = empty / len(df) * 100
            if empty > 0:
                report.append(f"| {name} | {col} | {empty:,} | {pct:.2f}% |")
    report.append("")

    # ---- Ground truth stats ----
    report.append("## 4. Ground Truth Statistics\n")
    
    # Parse matched IDs
    gt_matches = {}
    all_matched_ids = []
    s2_count = 0
    s3_count = 0
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"].strip()
        if matched == "":
            gt_matches[s1_id] = []
        else:
            ids = matched.split(",")
            gt_matches[s1_id] = ids
            all_matched_ids.extend(ids)
            for mid in ids:
                if mid.startswith("S2-"):
                    s2_count += 1
                elif mid.startswith("S3-"):
                    s3_count += 1

    n_singleton = sum(1 for v in gt_matches.values() if len(v) == 0)
    n_matched = sum(1 for v in gt_matches.values() if len(v) > 0)
    match_counts = [len(v) for v in gt_matches.values()]
    
    report.append(f"- **Total S1 entities:** {len(gt_matches):,}")
    report.append(f"- **Singletons (no matches):** {n_singleton:,} ({n_singleton/len(gt_matches)*100:.1f}%)")
    report.append(f"- **Non-singletons:** {n_matched:,} ({n_matched/len(gt_matches)*100:.1f}%)")
    report.append(f"- **Total matched S2 IDs:** {s2_count:,}")
    report.append(f"- **Total matched S3 IDs:** {s3_count:,}")
    report.append(f"- **Match count stats:** min={min(match_counts)}, max={max(match_counts)}, mean={np.mean(match_counts):.2f}, median={np.median(match_counts):.1f}")
    report.append("")

    print(f"\n  GT: {len(gt_matches):,} S1 entities, {n_singleton:,} singletons ({n_singleton/len(gt_matches)*100:.1f}%)")
    print(f"  Match count stats: min={min(match_counts)}, max={max(match_counts)}, mean={np.mean(match_counts):.2f}")

    # Distribution of match counts
    report.append("### Match Count Distribution\n")
    mc_counter = Counter(match_counts)
    report.append("| # Matches | # S1 Entities | % |")
    report.append("|-----------|---------------|---|")
    for k in sorted(mc_counter.keys()):
        report.append(f"| {k} | {mc_counter[k]:,} | {mc_counter[k]/len(gt_matches)*100:.2f}% |")
    report.append("")

    # ---- Verify deduplication assumption ----
    report.append("## 5. Deduplication Assumption Check\n")
    id_to_s1 = defaultdict(list)
    for s1_id, mids in gt_matches.items():
        for mid in mids:
            id_to_s1[mid].append(s1_id)
    
    multi_assigned = {mid: s1s for mid, s1s in id_to_s1.items() if len(s1s) > 1}
    report.append(f"- S2/S3 IDs assigned to >1 S1 entity: **{len(multi_assigned)}**")
    if multi_assigned:
        examples = list(multi_assigned.items())[:5]
        for mid, s1s in examples:
            report.append(f"  - `{mid}` -> {s1s}")
    else:
        report.append("- **CONFIRMED:** Each S2/S3 record belongs to at most one S1 entity.")
    report.append("")
    print(f"  Multi-assigned S2/S3 IDs: {len(multi_assigned)}")

    # ---- Do matches share country? ----
    report.append("## 6. Cross-Country Matches\n")
    s1_country = dict(zip(s1_train["entity_id"], s1_train["country"]))
    s2_country = dict(zip(s2_train["entity_id"], s2_train["country"]))
    s3_country = dict(zip(s3_train["entity_id"], s3_train["country"]))

    cross_country = 0
    same_country = 0
    for s1_id, mids in gt_matches.items():
        c1 = s1_country.get(s1_id, "")
        for mid in mids:
            c2 = s2_country.get(mid, "") or s3_country.get(mid, "")
            if c1 and c2:
                if c1 != c2:
                    cross_country += 1
                else:
                    same_country += 1
    
    report.append(f"- Same-country matches: {same_country:,}")
    report.append(f"- Cross-country matches: {cross_country:,}")
    if cross_country > 0:
        report.append(f"- **WARNING:** {cross_country} cross-country matches found!")
    else:
        report.append("- **CONFIRMED:** All matches are within the same country.")
    report.append("")
    print(f"  Cross-country matches: {cross_country}")

    # ---- Country string variants ----
    report.append("## 7. Country String Variants\n")
    all_countries = set()
    for df in [s1_train, s2_train, s3_train, s1_test, s2_test, s3_test]:
        all_countries.update(df["country"].unique())
    report.append(f"All unique country labels: `{sorted(all_countries)}`\n")
    print(f"  All country labels: {sorted(all_countries)}")

    # ---- Name/address noise examples ----
    report.append("## 8. Matched Group Examples\n")
    
    # Build lookup
    all_records = {}
    for df in [s1_train, s2_train, s3_train]:
        for _, row in df.iterrows():
            all_records[row["entity_id"]] = {
                "name": row["business_name"],
                "addr": row["business_address"],
                "country": row["country"]
            }

    # Get some non-singleton groups per country
    us_groups = []
    india_groups = []
    for s1_id, mids in gt_matches.items():
        if len(mids) == 0:
            continue
        c = s1_country.get(s1_id, "")
        if c == "US" and len(us_groups) < 15:
            us_groups.append((s1_id, mids))
        elif c == "India" and len(india_groups) < 15:
            india_groups.append((s1_id, mids))
        if len(us_groups) >= 15 and len(india_groups) >= 15:
            break

    for country, groups in [("US", us_groups), ("India", india_groups)]:
        report.append(f"### {country} Examples\n")
        for s1_id, mids in groups[:13]:
            r1 = all_records.get(s1_id, {})
            report.append(f"**S1 `{s1_id}`:** `{r1.get('name','')}` | `{r1.get('addr','')}`")
            for mid in mids[:4]:
                r = all_records.get(mid, {})
                report.append(f"  - `{mid}`: `{r.get('name','')}` | `{r.get('addr','')}`")
            report.append("")

    # ---- Noise catalogue ----
    report.append("## 9. Noise Patterns Observed\n")
    report.append("(Populated from examining examples above)\n")
    report.append("- Legal suffix variants: Corp/Corporation, Inc/Incorporated, LLC, Ltd/Limited, Pvt/Private")
    report.append("- Abbreviations: & vs 'and', Rd/Road, St/Street, Ave/Avenue, Blvd/Boulevard")
    report.append("- Hindi/Devanagari vs English transliteration in S2 names")
    report.append("- Missing address components (PIN codes, state names)")
    report.append("- Landmark references ('Near SBI ATM', 'Opp. Bus Stand')")
    report.append("- Casing variations (all caps, title case, mixed)")
    report.append("- Punctuation: periods, commas, hyphens in names")
    report.append("- Word order transpositions in addresses")
    report.append("- DBA / trade names vs legal names")
    report.append("")

    # ---- Runtime estimation ----
    report.append("## 10. Scale & Runtime Estimation\n")
    total_test = len(s1_test) + len(s2_test) + len(s3_test)
    report.append(f"- Test total records: {total_test:,}")
    report.append(f"- S1 (queries): {len(s1_test):,}")
    report.append(f"- S2+S3 (candidates): {len(s2_test)+len(s3_test):,}")
    report.append(f"- Naive all-pairs: {len(s1_test) * (len(s2_test)+len(s3_test)):,} (IMPOSSIBLE)")
    report.append(f"- With blocking (100 candidates/S1): ~{len(s1_test)*100:,} pairs to score")
    report.append("")

    # Save report
    report_path = os.path.join(REPORTS_DIR, "eda.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\n  EDA report saved to {report_path}")

    # ---- Quick stats for pipeline design ----
    print("\n" + "=" * 60)
    print("KEY NUMBERS FOR PIPELINE DESIGN:")
    print(f"  Train S1: {len(s1_train):,}")
    print(f"  Train S2+S3: {len(s2_train)+len(s3_train):,}")
    print(f"  Test S1: {len(s1_test):,}")
    print(f"  Test S2+S3: {len(s2_test)+len(s3_test):,}")
    print(f"  Singleton rate: {n_singleton/len(gt_matches)*100:.1f}%")
    print(f"  Multi-assigned: {len(multi_assigned)}")
    print(f"  Cross-country: {cross_country}")
    print("=" * 60)

    return {
        "train_s1": len(s1_train),
        "train_s2": len(s2_train),
        "train_s3": len(s3_train),
        "test_s1": len(s1_test),
        "test_s2": len(s2_test),
        "test_s3": len(s3_test),
        "singleton_rate": n_singleton / len(gt_matches),
        "multi_assigned": len(multi_assigned),
        "cross_country": cross_country,
    }


if __name__ == "__main__":
    run_eda()
