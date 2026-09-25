#!/bin/bash
set -euo pipefail

TEAM_NAME="Antigravity"
ZIP_NAME="${TEAM_NAME}_submission.zip"

echo "=========================================================="
echo "Packaging submission archive: ${ZIP_NAME}"
echo "=========================================================="

# Check that output files exist
if [ ! -f "output/matching_results.tsv" ] || [ ! -f "output/candidate_pairs.tsv" ]; then
    echo "ERROR: output/matching_results.tsv or output/candidate_pairs.tsv missing!"
    exit 1
fi

# Remove older zip if present
rm -f "${ZIP_NAME}"

# Create zip with exact structure required:
# output/matching_results.tsv
# output/candidate_pairs.tsv
# code/business_entity_resolution/src/...
# code/business_entity_resolution/README.md
# code/business_entity_resolution/requirements.txt
# Documentation_template.md

zip -r "${ZIP_NAME}" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/src/ \
    code/business_entity_resolution/README.md \
    code/business_entity_resolution/requirements.txt \
    Documentation_template.md \
    -x "*.pyc" -x "*__pycache__*" -x "*.DS_Store*" -x "*.pkl" -x "*.log"

echo ""
echo "Archive created successfully:"
ls -lh "${ZIP_NAME}"

echo ""
echo "Archive contents preview:"
unzip -l "${ZIP_NAME}" | head -25
echo "=========================================================="
