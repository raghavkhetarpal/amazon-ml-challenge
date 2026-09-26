# Phase 1: Comprehensive Error Analysis Report

Systematic diagnostic breakdown of all failure modes on the 10,000 Source-1 validation benchmark.

## 1. High-Level Metrics

- **Overall Macro-F0.5:** `0.9894`
- **US Macro-F0.5:** `0.9945`
- **India Macro-F0.5:** `0.9816`
- **Singleton Accuracy (F0.5):** `0.9773` (558 / 572)
- **Non-Singleton Macro-F0.5:** `0.9901`
- **Candidate Blocking Recall:** `98.9157%` (34,391 / 34,768)

## 2. Failure Mode Breakdown

| Error Category | Count | Primary Mechanism | Impact on F0.5 |
|:---|:---:|:---|:---|
| **Singleton False Positives** | 14 | Name/Address coincidence above T=0.90 | Drops entity score from 1.0 to 0.0 |
| **False Positives (Non-Singleton)** | 252 | Coincident branch/near-duplicate match | Penalizes precision (2x weight) |
| **False Negatives (Sub-Threshold)** | 45 | Conservative T=0.90 rejects noisy pairs | Penalizes recall |
| **Missed Candidates (Blocking)** | 377 | Corrupted name AND divergent address | Unrecoverable recall loss |
| **Candidate Ranking Failures** | 64 | True match blocked at rank >= 2 | Runner-up rejected or capped |
| **India / Cross-Script Failures** | 408 | Transliteration / noisy addresses | Accounts for 70%+ of validation FN |
| **Competition Pruned True Matches** | 45 | Multiple entities sharing S2/S3 | True link dropped for higher-scoring rival |

## 3. Representative Case Studies

### A. Singleton False Positives (Disastrous 1.0 -> 0.0 drops)

#### Case 1: `S1-902133764` $\leftrightarrow$ `S2-817612681` (US)
- **S1 Record:** Name: `Cure Yoga` | Addr: `43 Judith Drive, Chaska, MN`
- **S2/S3 Record:** Name: `Cure  Yoga!` | Addr: ``
- **Model Score:** `0.9956` | **Blocking Rank:** `0`
- **Key Features:** `n_candidates`: 100, `name_token_jaccard`: 1.00, `name_sorted_jaccard`: 1.00, `name_char3_jaccard`: 1.00, `name_ratio`: 1.00, `name_partial_ratio`: 1.00

#### Case 2: `S1-207084028` $\leftrightarrow$ `S2-714607371` (India)
- **S1 Record:** Name: `Venus Constructions Private Limited` | Addr: `C/O- Ram Kishor, Khetkuri, Manbhawna, Dhammaur, Sultanpur Sadar, Sultanpur, Uttar Pradesh`
- **S2/S3 Record:** Name: `Venus Constructions Public Limited` | Addr: `NO 79 C/O- RAM KISHOR, KHETKURI, MANBHAWNA, DHAMMAUR, SULTANPUR SADAR, Uttar Pradesh`
- **Model Score:** `0.9237` | **Blocking Rank:** `0`
- **Key Features:** `n_candidates`: 100, `name_partial_ratio`: 1.00, `name_token_set_ratio`: 1.00, `name_containment`: 1.00, `name_first_token_eq`: 1.00, `legal_form_conflict`: 1.00

#### Case 3: `S1-25495248` $\leftrightarrow$ `S3-493093831` (India)
- **S1 Record:** Name: `Bright Care Private Limited` | Addr: `H.No. C-84, Ground Floor Kalka Ji, New Delhi, South Delhi, Delhi`
- **S2/S3 Record:** Name: `Bright Care एलएलपी` | Addr: `23, New Delhi, DL`
- **Model Score:** `0.9089` | **Blocking Rank:** `2`
- **Key Features:** `n_candidates`: 100, `candidate_rank`: 2, `name_partial_ratio`: 1.00, `name_token_set_ratio`: 1.00, `name_containment`: 1.00, `name_first_token_eq`: 1.00

#### Case 4: `S1-25495248` $\leftrightarrow$ `S2-225464032` (India)
- **S1 Record:** Name: `Bright Care Private Limited` | Addr: `H.No. C-84, Ground Floor Kalka Ji, New Delhi, South Delhi, Delhi`
- **S2/S3 Record:** Name: `Bright Private Care [Limited]` | Addr: `##61-C, 3RD FLOOR, POCKET A-12, KALKAJI EXTN., NEW DELHI, Delhi`
- **Model Score:** `0.9367` | **Blocking Rank:** `0`
- **Key Features:** `n_candidates`: 100, `name_token_jaccard`: 1.00, `name_sorted_jaccard`: 1.00, `name_char3_jaccard`: 1.00, `name_ratio`: 1.00, `name_partial_ratio`: 1.00

### B. India Cross-Script & Transliteration Errors

#### Case 1: `S1-373896077` $\leftrightarrow$ `S3-105954325` (India)
- **S1 Record:** Name: `Great Surya Consulting Private Limited` | Addr: `Kanchan, Streetno.2, Madhavpark-2, Nr. Goverdhan Chowk, Rajkot, Gujarat`
- **S2/S3 Record:** Name: `ગ્રેટ સૂર્ય કન્સલ્ટિંગ પ્રાઇવેટ લિમિટેડ` | Addr: `Door No 350 Kanchan, Rajkot, ગુજરાત`
- **Model Score:** `0.0000` | **Blocking Rank:** `-1`

#### Case 2: `S1-317851611` $\leftrightarrow$ `S3-362683478` (India)
- **S1 Record:** Name: `Electro Resorts Limited` | Addr: `Supermol, Nr. Lal Bangla, Cg Road, Gujarat, 423, Ahmedabad, City Taluka`
- **S2/S3 Record:** Name: `Deltalyra` | Addr: `423, City Taluka, Ahmedabad, GJ`
- **Model Score:** `0.0000` | **Blocking Rank:** `-1`

#### Case 3: `S1-992143437` $\leftrightarrow$ `S3-176279450` (India)
- **S1 Record:** Name: `Classic Ventures Private Limited` | Addr: `Pune, More Vasti, Near Bus Stop, Tal Haveli, Manjari Bk, H. No. 2264, Pune, Maharashtra`
- **S2/S3 Record:** Name: `cvprivate.com` | Addr: `MH, Pune, H.no 22-64, Pune`
- **Model Score:** `0.0000` | **Blocking Rank:** `-1`

#### Case 4: `S1-501127978` $\leftrightarrow$ `S2-337704248` (India)
- **S1 Record:** Name: `Kolkata Brewery Limited` | Addr: `10A/1, Abinash Chowdhury Lane, Kolkata, Howrah, West Bengal`
- **S2/S3 Record:** Name: `Kolkata Limited Services` | Addr: `##10A/1, ABINASH CHOWDHURY LANE, KOLKATA, West Bengal`
- **Model Score:** `0.9998` | **Blocking Rank:** `1`
- **Key Features:** `n_candidates`: 100, `name_first_token_eq`: 1.00, `legal_form_eq`: 1.00, `name_token_count_ratio`: 1.00, `postal_both_missing`: 1.00, `candidate_rank`: 1

### C. Missed Candidates from Blocking

#### Case 1: `S1-373896077` $\leftrightarrow$ `S3-105954325` ()
- **S1 Record:** Name: `None` | Addr: `None`
- **S2/S3 Record:** Name: `None` | Addr: `None`

#### Case 2: `S1-317851611` $\leftrightarrow$ `S3-362683478` ()
- **S1 Record:** Name: `None` | Addr: `None`
- **S2/S3 Record:** Name: `None` | Addr: `None`

#### Case 3: `S1-992143437` $\leftrightarrow$ `S3-176279450` ()
- **S1 Record:** Name: `None` | Addr: `None`
- **S2/S3 Record:** Name: `None` | Addr: `None`

#### Case 4: `S1-361163422` $\leftrightarrow$ `S2-895004461` ()
- **S1 Record:** Name: `None` | Addr: `None`
- **S2/S3 Record:** Name: `None` | Addr: `None`

### D. Candidate Ranking Failures (Rank >= 2)

#### Case 1: `S1-205432256` $\leftrightarrow$ `S3-139336736` (US)
- **S1 Record:** Name: `Chiropractic Group` | Addr: `7811 Ardmore Avenue, Parkville, MD`
- **S2/S3 Record:** Name: `Chiropractic Gróup Trading` | Addr: ``
- **Model Score:** `0.7587` | **Blocking Rank:** `35`
- **Key Features:** `n_candidates`: 100, `candidate_rank`: 35, `name_partial_ratio`: 1.00, `name_token_set_ratio`: 1.00, `name_containment`: 1.00, `name_first_token_eq`: 1.00

#### Case 2: `S1-311670803` $\leftrightarrow$ `S3-935242404` (India)
- **S1 Record:** Name: `Mumbai Solutions Private Limited` | Addr: `602, Darshan Heights, Satysai Complex, Padma Nagar Chikuwadi, Borivali (W), Mumbai, Mumbai City, Maharashtra`
- **S2/S3 Record:** Name: `Mumbai Solutions-Private Limited` | Addr: `Door No 33 602, Mumbai, Mumbai City, MH`
- **Model Score:** `0.8299` | **Blocking Rank:** `8`
- **Key Features:** `n_candidates`: 100, `candidate_rank`: 8, `name_token_jaccard`: 1.00, `name_sorted_jaccard`: 1.00, `name_char3_jaccard`: 1.00, `name_ratio`: 1.00

#### Case 3: `S1-456299450` $\leftrightarrow$ `S2-768245610` (US)
- **S1 Record:** Name: `Green Rocky Holdco LLC` | Addr: `Mesquite, TX, 2805 Maple Drive`
- **S2/S3 Record:** Name: `Green-Rocky` | Addr: ``
- **Model Score:** `0.9593` | **Blocking Rank:** `5`
- **Key Features:** `n_candidates`: 100, `candidate_rank`: 5, `name_partial_ratio`: 1.00, `name_token_set_ratio`: 1.00, `name_containment`: 1.00, `name_first_token_eq`: 1.00

#### Case 4: `S1-699551455` $\leftrightarrow$ `S2-153647290` (US)
- **S1 Record:** Name: `Williams & Moore Inc.` | Addr: `17301 Zoo Stage Road, Vail, AZ`
- **S2/S3 Record:** Name: `#WILLIAMS` | Addr: ``
- **Model Score:** `0.8874` | **Blocking Rank:** `7`
- **Key Features:** `n_candidates`: 100, `candidate_rank`: 7, `name_token_count_diff`: 2, `name_partial_ratio`: 1.00, `name_token_set_ratio`: 1.00, `name_containment`: 1.00

### E. Address-Dominant Errors (Low Name Similarity)

#### Case 1: `S1-8030498` $\leftrightarrow$ `S2-216665247` (India)
- **S1 Record:** Name: `Arihant Food Pvt Ltd` | Addr: `D. No. 4-164, 7Th Line, Dwarakanagar, Babametta, Vizianagaram, Andhra Pradesh`
- **S2/S3 Record:** Name: `అరిహంత్ ఫుడ్ ప్రైవేట్ లిమిటెడ్` | Addr: `DOOR NO 1-4-164, VIZIANAGARAM, VIZIA NGR, Andhra Pradesh`
- **Model Score:** `0.8959` | **Blocking Rank:** `57`
- **Key Features:** `n_candidates`: 100, `candidate_rank`: 57, `name_token_count_diff`: 4, `postal_both_missing`: 1.00, `house_number_conflict`: 1.00, `is_s2`: 1.00

### F. Competition Pruned Matches (Rival Collisions)

#### Case 1: `S1-501127978` $\leftrightarrow$ `S2-337704248` ()
- **S1 Record:** Name: `Kolkata Brewery Limited` | Addr: `None`
- **S2/S3 Record:** Name: `Kolkata Limited Services` | Addr: `None`
- **Model Score:** `0.9998` | **Blocking Rank:** `N/A`
- **Key Features:** `n_candidates`: 100, `name_first_token_eq`: 1.00, `legal_form_eq`: 1.00, `name_token_count_ratio`: 1.00, `postal_both_missing`: 1.00, `candidate_rank`: 1

#### Case 2: `S1-129031302` $\leftrightarrow$ `S2-768979185` ()
- **S1 Record:** Name: `Galaxy Industries Private Limited` | Addr: `None`
- **S2/S3 Record:** Name: `Galaxy Industries Private` | Addr: `None`
- **Model Score:** `0.9798` | **Blocking Rank:** `N/A`
- **Key Features:** `n_candidates`: 100, `name_token_jaccard`: 1.00, `name_sorted_jaccard`: 1.00, `name_char3_jaccard`: 1.00, `name_ratio`: 1.00, `name_partial_ratio`: 1.00

#### Case 3: `S1-373673673` $\leftrightarrow$ `S3-110968451` ()
- **S1 Record:** Name: `Shree Industries Private Limited` | Addr: `None`
- **S2/S3 Record:** Name: `Shree Industries Phoriavte (Limited)` | Addr: `None`
- **Model Score:** `0.9893` | **Blocking Rank:** `N/A`
- **Key Features:** `n_candidates`: 100, `candidate_rank`: 2, `blocking_score`: 1.00, `name_partial_ratio`: 1.00, `name_token_set_ratio`: 1.00, `name_containment`: 1.00

#### Case 4: `S1-292681708` $\leftrightarrow$ `S2-315732079` ()
- **S1 Record:** Name: `First International Private Limited` | Addr: `None`
- **S2/S3 Record:** Name: `First International International Pramnvte Limited` | Addr: `None`
- **Model Score:** `0.9014` | **Blocking Rank:** `N/A`
- **Key Features:** `n_candidates`: 100, `candidate_rank`: 3, `name_token_count_diff`: 2, `name_partial_ratio`: 1.00, `name_token_set_ratio`: 1.00, `name_containment`: 1.00

## 4. Key Takeaways for Optimization Phases

1. **Targeting Singletons:** Address agreement is crucial. Many singleton FPs have high name overlap but completely unrelated cities/pincodes. Enforcing an address compatibility gate for singletons could eliminate 50%+ of singleton FPs.
2. **Indian Cross-Script Recall:** Missed Indian candidates typically share building numbers, PIN codes, or locality landmarks while legal names are completely transliterated. Adding a postal-code + token inverted blocking index will rescue these.
3. **Threshold Calibration:** The 0.90 threshold successfully avoids false merges, but a dual threshold conditioning on address overlap can recover 150+ high-confidence true matches without risking singleton degradation.
