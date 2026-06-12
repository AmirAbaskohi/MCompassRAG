#!/usr/bin/env bash
set -euo pipefail
mkdir -p third_party data outputs
clone () { local url="$1" dst="$2"; if [ ! -d "$dst/.git" ]; then git clone --depth 1 "$url" "$dst"; else echo "exists: $dst"; fi; }
clone https://github.com/AmirAbaskohi/CEMTM.git  third_party/CEMTM
clone https://github.com/Fitz-like-coding/CWTM.git third_party/CWTM
clone https://github.com/adjidieng/ETM.git        third_party/ETM
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
echo "Setup complete."
