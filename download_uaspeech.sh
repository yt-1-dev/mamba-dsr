#!/bin/sh
# Download the noise-reduced UASpeech dysarthria set into ../data (matches preprocess.py paths).
#
# The original UASpeech corpus is licence-gated (request access from UIUC:
# http://www.isle.illinois.edu/sst/data/UASpeech/). This pulls the mirrored
# noise-reduced version from Kaggle instead; override the slug if it moves.
set -e
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-../data}"
SLUG="${KAGGLE_DATASET:-aryashah2k/noise-reduced-uaspeech-dysarthria-dataset}"
DEST="$DATA_DIR/Noise Reduced UASPEECH Dysarthria Dataset"

if [ -d "$DEST/noisereduced-uaspeech" ]; then
    echo "already present: $DEST"
    exit 0
fi

command -v kaggle >/dev/null || { echo "need the kaggle CLI: pip install kaggle, then put your API token in ~/.kaggle/kaggle.json"; exit 1; }

mkdir -p "$DEST"
kaggle datasets download -d "$SLUG" -p "$DATA_DIR" --unzip -o
echo "done: $DEST"
