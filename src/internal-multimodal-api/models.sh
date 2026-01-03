#!/usr/bin/env bash
set -e

# paths
BASE=/workspace/dataspace/internal-multimodal-api/models

mkdir -p $BASE

echo "[1] Download Whisper Large V3 (CTranslate2)"
pip install ctranslate2 transformers huggingface_hub -q

ct2-transformers-converter \
  --model openai/whisper-large-v3 \
  --output_dir $BASE/whisper-large-v3-ct2 \
  --copy_files tokenizer.json \
  --quantization bfloat16

echo "[2] Download pyannote diarization 3.1"

huggingface-cli download pyannote/speaker-diarization-3.1 \
  --token "<token>" \
  --local-dir $BASE/pyannote-diarization-3.1 \
  --local-dir-use-symlinks False

echo "Done."