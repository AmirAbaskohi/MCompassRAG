#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/train_retriever.yaml}"
python -m src.training.train --config "$CONFIG"
