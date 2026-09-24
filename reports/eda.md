# EDA Report: Business Entity Resolution

## 1. Dataset Scale

### Row Counts

| Source | Train | Test |
|--------|-------|------|
| S1 | 2,206,821 | 1,732,544 |
| S2 | 5,034,616 | 4,887,273 |
| S3 | 5,285,603 | 5,082,316 |
| Ground Truth | 2,206,821 | N/A |
| **Total** | **12,527,040** | **11,702,133** |

### Country Distribution

**Train S1:** {'US': np.int64(1323633), 'India': np.int64(883188)}

**Train S2:** {'US': np.int64(3016817), 'India': np.int64(2017799)}

**Train S3:** {'US': np.int64(3170056), 'India': np.int64(2115547)}

**Test S1:** {'India': np.int64(809986), 'US': np.int64(663106), 'France': np.int64(259452)}

**Test S2:** {'India': np.int64(2312565), 'US': np.int64(1871330), 'France': np.int64(703378)}

**Test S3:** {'India': np.int64(2405000), 'US': np.int64(1945701), 'France': np.int64(731615)}

## 2. ID Formats

**Train S1:** e.g. `['S1-925783039', 'S1-773889195', 'S1-377745466']`
**Train S2:** e.g. `['S2-166376419', 'S2-764573417', 'S2-639257739']`
**Train S3:** e.g. `['S3-202863386', 'S3-859268022', 'S3-22467283']`

## 3. Missing / Empty Fields

| Source | Field | Empty Count | % |
|--------|-------|-------------|---|
| Train S2 | business_address | 168,967 | 3.36% |
| Train S3 | business_address | 175,916 | 3.33% |
| Test S2 | business_address | 129,408 | 2.65% |
| Test S3 | business_address | 136,098 | 2.68% |

## 4. Ground Truth Statistics

- **Total S1 entities:** 2,206,821
- **Singletons (no matches):** 123,247 (5.6%)
- **Non-singletons:** 2,083,574 (94.4%)
- **Total matched S2 IDs:** 3,693,619
- **Total matched S3 IDs:** 3,944,746
- **Match count stats:** min=0, max=11, mean=3.46, median=3.0

### Match Count Distribution

| # Matches | # S1 Entities | % |
|-----------|---------------|---|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |

## 5. Deduplication Assumption Check

- S2/S3 IDs assigned to >1 S1 entity: **0**
- **CONFIRMED:** Each S2/S3 record belongs to at most one S1 entity.

## 6. Cross-Country Matches

- Same-country matches: 7,638,365
- Cross-country matches: 0
- **CONFIRMED:** All matches are within the same country.

## 7. Country String Variants

All unique country labels: `['France', 'India', 'US']`

## 8. Matched Group Examples

### US Examples

**S1 `S1-965667`:** `Maure Williams Colombier Inc` | `85 Wayne Avenue, Ticonderoga, NY`
  - `S2-681193310`: `Maure Wilblims Colombier Inc` | ``
  - `S2-743505751`: `Maure Williams Colombier` | ``
  - `S3-775321672`: `Dréxkor` | `85 Wanye Avenue, Ticonderoga Townshiip, New York`
  - `S3-11291185`: `maurewilliamscolombier.com` | `Wayne Ave, Ticonderoga Townshiip, New York`

**S1 `S1-343815751`:** `Dahlia Power Reliable Scientific LLC` | `630 45th Terrace, Kansas City, MO`
  - `S2-790675320`: `Dahlia Power Reliable` | `KANSAS CITY, MO, 630 45ND TERRACE, null`
  - `S2-479876582`: `Dahlia Power Reliable Scientific` | `45ND TERRACE, null, KANSAS CITY, MO`
  - `S3-878454467`: `Dahlia Ponr Reliable Scientific LLC` | `Missouri, 630 45th Terrace, Kansas City`

**S1 `S1-102811957`:** `Payne Enterprises` | `3315 Fremont Street, Peoria, IL`
  - `S2-478959098`: `Payne Énterprises` | `3315 FREMONT ST, PEORIA, IL`
  - `S2-553508714`: `Payne Enterpires` | `3315 FREMONT ST, PEORIA, IL`
  - `S2-625774905`: `PAYNE-ENRTPRMISES` | `3315 FREMONT SAINT, PEORIA, IL`
  - `S3-728090388`: `Payne Etrepndiels` | `3315 Fremont St, Peoria, Illinois`

**S1 `S1-18727616`:** `Lumay Boral` | `1056 Belden Avenue, Akron, OH`
  - `S2-755677256`: `Lumay Boral Inc.` | `1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH`
  - `S3-187831601`: `Lumay Bóral` | `1056c Belden Ave, AKON, Ohio`
  - `S3-641489370`: `Lumay Boral` | `1056c Belden Ave, AKON, Ohio`
  - `S3-476250621`: `Lumay Bóral` | `1056c Belden Avenue, AKON, Ohio`

**S1 `S1-29845983`:** `Hendricks and Flowers Inc` | `33 Sleepy Hollow Drive, Danbury, CT`
  - `S2-648035184`: `Hendricks and  Flowers Inc` | `CT, SLEEPY HOLLOW DRIVE, DANBURY`
  - `S3-588502663`: `Hendricks and Inc Flowers` | ``

**S1 `S1-730934468`:** `Orellana Investments LLC` | `728 A Quail Avenue, Fl Ground Floor, Geneva, IA`
  - `S2-356983532`: `Orellana Investments Investments Llc` | ``
  - `S3-352439310`: `LLC Orellana Invsmbens` | `728 A Quail Avenue, Fl. Ground Floor, Geneva, Iowa`

**S1 `S1-546142636`:** `Crystal Staffing Solutions LLC` | `8706 Kentucky Derby Drive, Waxhaw, NC`
  - `S2-487600131`: `Crystal Solutions LLC Partners` | `8706 KENTUCKY DERBY DR, WAXHAW, NC`
  - `S2-582477216`: `CRYSTAL STAFFING SOLUTIONS-L.L.C.` | `8706 KENTUCKY DERBY DR, WAXHAW, NC`
  - `S2-392804085`: `LLC Crystal Sttfrifng Solutions` | `8706 KENTUCKY DERBY DRIVE, WAXHAW, NC`
  - `S3-200008747`: `Llc Crystal Staffing Solutions` | `870 Kentucky Derby Drive, Waxhaw, North Carolina`

**S1 `S1-274126313`:** `Obsidian, LLC` | `3907 Hamilton Road, Deer Park, WA`
  - `S2-736616474`: `Obsidian,-LLC` | `3907 HAMILTON RD, DEER PARK CIYT, WA`
  - `S2-680265918`: `obsidian, llc` | ``
  - `S2-51486805`: `Obsidian, LLC Center` | `3907 HAMILTON RD, DEER PARK CIYT, WA`
  - `S3-925631694`: `Obsidian, Llc` | `Deer Park, Washington, Hamilton Rd`

**S1 `S1-145361722`:** `Dick Regional Armada Corp` | `33466 Warwick Hills Road, Yucaipa, CA`
  - `S2-120366543`: `Dick Regional` | ``
  - `S2-939389287`: `Dick Regional Armada` | `33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA`
  - `S3-96572514`: `[Corp] Dick Regional Armada` | `Yucaipa, California, 33466 Warwik Hills Road`

**S1 `S1-503957000`:** `AP Hospitality Inc` | `20085 Us 23, Circleville, OH`
  - `S2-994658326`: `AP INC SERVICE #80430` | `20085 US 23, CIRCLEVILLE, OH`
  - `S2-235117490`: `AP Hospitality Ínc` | `20085 US 23, CIRCLEVILLE, OH`
  - `S2-353308450`: `AP  Inc Hospitality` | `20085 US 23, CIRCLEVILLE, OH`
  - `S2-173295926`: `AP Ínc Center` | `20085 US 23, CIRCLEVILLE, OH`

**S1 `S1-692000596`:** `Desai, Alanah, Esq., PC` | `584 Summer Street, Unit 1441, Holyoke, MA`
  - `S2-580419223`: `DESAI, ALANAH,-ESQ.,` | `SUMMER STREET, HOLLYOKE, MA`
  - `S3-164220452`: `Desai, Alanah, Ésq.,` | `584 Summer Street, Holyoke, Massachusetts`

**S1 `S1-777597828`:** `Uptown Pub` | `6114 10th Avenue, Spokane Valley, WA`
  - `S2-836452886`: `UPTOWN  PUB CO` | `6114 10RD AVENUE, SPOKANE VALLEY, WA`
  - `S3-413669121`: `Solkeloquo` | `6114 Tenth Ave, Spokane Valley, Washington`

**S1 `S1-65263544`:** `Caldeon Nova` | `1 Greenbrier Drive, Kimberling City, MO`
  - `S2-881122703`: `Caldeon  Nova` | `GREENBRIER DRIVE, KIMBERLING CITY, MO`
  - `S2-998270769`: `CALDEON NOVA LLC` | ``
  - `S2-180463458`: `CALDEON NOVA` | `MO, KIMBERLING CITY, 1 GREENBRIER DRIVE`
  - `S3-403476502`: `Caldeon Nova` | `#1 Greenbrier Drive, Kimberling City, Missouri`

### India Examples

**S1 `S1-55344266`:** `Raj Investments LLP` | `6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu`
  - `S2-249013014`: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu`
  - `S2-197070651`: `Raj Investments LLP` | `6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu`
  - `S3-478195123`: `Raj Investments எல்எல்பி` | `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN`
  - `S3-384364074`: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | `6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு`

**S1 `S1-656753428`:** `Ss Food Private Limited` | `Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh`
  - `S2-153058913`: `एसएस फूड प्राइवेट लिमिटेड` | `AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश`
  - `S2-24659151`: `एसएस फूड प्राइवेट लिमिटेड` | `AF-0684, Uttar Pradesh, GHAZIABAD, 9487203`
  - `S3-679606215`: `एसएस फूड प्राइवेट लिमिटेड` | `Af-684, Ghaziabad, UP`

**S1 `S1-318373630`:** `Red Ventures Private Limited` | `Rajasthan, Jaipur, Banipark, Gokul Apartment, E-3A Kanti Chandra Road, G-1`
  - `S2-660036492`: `रेड वेंचर्स प्राइवेट लिमिटेड` | `G-1, BANIPARK, JAIPUR, Rajasthan`
  - `S3-804600254`: `Red Ventures Private` | `Doro No 316 G-1, Gokul Apartment, E-3a Kanti Chandra Road, Banipark, Subhash Nagar, RJ`

**S1 `S1-86989137`:** `Laxmi Golden Investments Private Limited` | `New Bridge Business Centre'S 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, Karnataka`
  - `S3-274817120`: `Laxmi Gbn lnvestments Private Limited` | `New Bridge Bssiness Centre's 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, ಕರ್ನಾಟಕ`
  - `S3-312496301`: `Laxmi Golden Investments` | `New Bridge Buisness Centre's 11Th Floor, N1 Block Embassy Manyata Business Tech Park, Naga, Wara, Bangalore, KA`

**S1 `S1-789009573`:** `Hotel Enterprises Limited` | `Wz-187C Shop No.13, 14 Kh. No.47 S/F. Vikaspuri Budhela Village Behind Oxford School, Delhi, West Delhi, Delhi`
  - `S2-383871912`: `होटल एंटरप्राइजेज लिमिटेड` | `WZ-187C SHOP NO.13, DELHI, WEST DELHI, Delhi`
  - `S3-74481402`: `Hotel Limited Services` | `Block B-517 Wz-187c Shop No.13, Divreportingcircle, West Delhi, DL`
  - `S3-576451439`: `Hotel Énterprises Limited` | `Block B-517 Wz-187c Shop No.13, South West Delhi, Delhi, DL`

**S1 `S1-7293388`:** `Chordia & Partners` | `Faridabad, 1038 Sector 9, Haryana`
  - `S2-7028416`: `Chordia & Partners Company` | `हरियाणा, 1038 SECTOR 9, FARIDABAD`
  - `S2-442723188`: `Chordia + Pagnters - 7306204978` | `हरियाणा, DOOR NO 1038 SECTOR 9, FARIDABAD`
  - `S2-157073701`: `Chordia &-Pártners Ltd` | `H.NO 1038 SECTOR 9, FARIABAD, Haryana`
  - `S3-523120965`: `Smt Chordia  & Center` | `#1038 Sector 9, Faridabad, हरियाणा`

**S1 `S1-561341312`:** `Balaji Investment Private Limited` | `Plot No. D-88 & D-90, Hyd, Telangana, Hyderabad, Jeedimetla`
  - `S2-483615364`: `బాలాజీ ఇన్వెస్ట్‌మెంట్ ప్రైవేట్ లిమిటెడ్` | `H.NO 00516 PLOT NO. D-88 & D-90, JEEDIMETLA, HYD, HYDERABAD, Telangana`
  - `S3-619529814`: `Balaji Insvstemnt Private Limited` | `Plot No. D-88 & D-90, Hyd, Shapurnagar, TG`

**S1 `S1-282467635`:** `Prem & Sons Pvt Ltd` | `D-4, Chandana Apartments82, Infantry Road, Bangalore, Karnataka`
  - `S2-938895481`: `Prem & Pvt Ltd Services` | `D-4, CHANDANA APARTMENTS82, INFANTRY ROAD, Karnataka`
  - `S3-138350041`: `Prem & Pvt Ltd Partners` | `D-4, Bangalore, KA`

**S1 `S1-727602285`:** `Swastik Om Solutions LLP` | `201/D, Aditya, Svp Nagar Andheri (W), Mumbai, Mumbai City, Maharashtra`
  - `S2-976870196`: `स्वस्तिक ॐ सॉल्यूशंस एलएलपी` | `201/D, BHIWANDI, MUMBAI CITY, Maharashtra`
  - `S2-947230367`: `स्वस्तिक ॐ सॉल्यूशंस एलएलपी` | `201/D, ADITYA, SVP NAGAR ANDHERI (W), BHIWANDI, MUMBAI CITY, Maharashtra`
  - `S2-23141904`: `>> SWASTIK OM SOLUTIONS L.L.P.` | `201/D, BHIWANDI, BOMBAY, महाराष्ट्र`
  - `S2-138660620`: `[LLP] Swastik Om Solutions` | `201/D, BHIWANDI, BOMBAY, Maharashtra`

**S1 `S1-264156494`:** `Diksha Technologies Private Limited` | `A-401, Nitesh Central Par, Bagalur Main Road, Bangalore North, Bangalore, Karnataka`
  - `S2-184087846`: `Dr Diksha Private Limited Services` | `A-401, NITESH CENTRAL PAR, BAGALUR MAIN ROAD, BANGALORE NORTH, Karnataka`
  - `S3-562014765`: `Diksha Technologies Private Ltd` | `A-401, Nitesh Central Par, Bagalur Main Road, Bangalore, Bangalore North, KA`
  - `S3-544213330`: `Diksha Technologies Private` | `A-401, Nitesh Central Par, Bagalur Main Road, Bangalore North, Bangalore, KA`

**S1 `S1-525304403`:** `True Factory Ltd` | `C/O Sri G P Srivastva At, Idagah Mohall, Dehri-On-S, Dehri, Rohtas, Bihar`
  - `S2-815373751`: `True Factory-Ltd` | `C/O SRI G P SRIVASTVA AT, IDAGAH MOHALL, DEHRI-ON-S, DEHRI, Bihar`
  - `S2-293401164`: `truefactory.com` | `C/O SRI G P SRIVASTVA AT, IDAGAH MOHALL, DEHRI-ON-S, DEHRI, बिहार`
  - `S3-477957595`: `true factory ltd` | `C/o Sri G P Srivastva At, Idagah Mohall, Dehri-on-s, Dehri, Rohtas, BR`

**S1 `S1-314714647`:** `Seabird (India) Projects-Lucknow` | `Lucknow, 3/77, Lucknow, Vipul Khand, Opp. Study Hall School Gomtinagar, Uttar Pradesh`
  - `S2-968477409`: `Seabird (India)` | `3/77, VIPUL KHAND, OPP. STUDY HALL SCHOOL GOMTINAGAR, LUCKNOW, Uttar Pradesh`
  - `S2-46909251`: `Seabird (India) Projects-Lucknow` | `3/77, VIPUL KHAND, OPP. STUDY HALL SCHOOL GOMTINAGAR, LUCKNOW, Uttar Pradesh`
  - `S3-837108914`: `Mr Seabird (India) Projects-Lucknow LLP` | `3/7, Vipul Khand, Opp. Study Hall School Gomtinagar, Lucknow, UP`
  - `S3-192802051`: `Xylonexbrix` | `3/7, Lucknow, UP`

**S1 `S1-72310418`:** `Burger Solution Limited` | `Flat No. A702, 1St Main, Mantri Classic, 8Th Cross S.T. Bed Layout, Koramangala, Bangalore, Karnataka`
  - `S2-516305116`: `Sri Burger  Solution` | `DOOR NO 25 FLAT NO. A702, 1ST MAIN, MANTRI CLASSIC, 8TH CROSS S.T. BED LAYOUT, KORAMANGALA, BANGALORE, Karnataka`
  - `S2-11779506`: `Burger Solutino Limited` | `NO 25 FLAT NO. A702, 1ST MAIN, MANTRI CLASSIC, 8TH CROSS S.T. BED LAYOUT, KORAMANGALA, BANGALORE, Karnataka`
  - `S3-570980846`: `Burger Shutgon Limited` | `Flat No. A702, Bangalore, KA`
  - `S3-731160199`: `Burgersolution.Com` | `Flat No. A702, Bangalore, KA`

## 9. Noise Patterns Observed

(Populated from examining examples above)

- Legal suffix variants: Corp/Corporation, Inc/Incorporated, LLC, Ltd/Limited, Pvt/Private
- Abbreviations: & vs 'and', Rd/Road, St/Street, Ave/Avenue, Blvd/Boulevard
- Hindi/Devanagari vs English transliteration in S2 names
- Missing address components (PIN codes, state names)
- Landmark references ('Near SBI ATM', 'Opp. Bus Stand')
- Casing variations (all caps, title case, mixed)
- Punctuation: periods, commas, hyphens in names
- Word order transpositions in addresses
- DBA / trade names vs legal names

## 10. Scale & Runtime Estimation

- Test total records: 11,702,133
- S1 (queries): 1,732,544
- S2+S3 (candidates): 9,969,589
- Naive all-pairs: 17,272,751,604,416 (IMPOSSIBLE)
- With blocking (100 candidates/S1): ~173,254,400 pairs to score
