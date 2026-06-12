#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/rag/cemtm.yaml}"
QUERY="${2:-}"
python -m src.run --config "$CONFIG" --build-index
if [ -n "$QUERY" ]; then python -m src.run --config "$CONFIG" --query "$QUERY"; fi
