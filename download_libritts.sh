#!/bin/sh
# Download LibriTTS train-clean-100 into ../data/LibriTTS (matches preprocess.py paths).
set -e
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-../data}"
URL="https://www.openslr.org/resources/60/train-clean-100.tar.gz"

mkdir -p "$DATA_DIR"
if [ -d "$DATA_DIR/LibriTTS/train-clean-100" ]; then
    echo "already present: $DATA_DIR/LibriTTS/train-clean-100"
    exit 0
fi

curl -L -C - -o "$DATA_DIR/train-clean-100.tar.gz" "$URL"
tar -xzf "$DATA_DIR/train-clean-100.tar.gz" -C "$DATA_DIR"   # tarball already contains LibriTTS/
rm "$DATA_DIR/train-clean-100.tar.gz"
echo "done: $DATA_DIR/LibriTTS/train-clean-100"
