#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/rag/cemtm.yaml}"
python -m src.run --config "$CONFIG" --build-index
