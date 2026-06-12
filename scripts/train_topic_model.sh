#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/train_topic_model.yaml}"
python -m topic_models.train_topic_model --config "$CONFIG"
