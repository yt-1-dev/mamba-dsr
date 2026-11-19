#!/bin/sh
# Regenerate python_files/ from the notebooks. Run after editing any .ipynb.
# --to python (not --to script): "script" picks the extension from each notebook's
# language_info metadata and falls back to .txt when it's missing.
cd "$(dirname "$0")" && exec jupyter nbconvert --to python *.ipynb --output-dir python_files
