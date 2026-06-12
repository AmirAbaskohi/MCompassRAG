#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/gen_train_data.yaml}"
python -m data_gen.build_training_data --config "$CONFIG"
