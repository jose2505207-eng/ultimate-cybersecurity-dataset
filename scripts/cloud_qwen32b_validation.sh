#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config/qlora_cloud_qwen25_coder_32b.yml}"
DATASET_PATH="${DATASET_PATH:-outputs/sft_dataset/qwen_cyber_sft_v2_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100}"
TRAIN_ROWS="${TRAIN_ROWS:-4096}"
EVAL_ROWS="${EVAL_ROWS:-512}"
MAX_STEPS="${MAX_STEPS:-100}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
EVAL_EVERY="${EVAL_EVERY:-25}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  LOCAL_FLAG=(--local-files-only)
else
  LOCAL_FLAG=()
fi

mkdir -p "${OUTPUT_DIR}"

echo "[cloud-qwen32b] Python: $(python --version)"
echo "[cloud-qwen32b] Config: ${CONFIG_PATH}"
echo "[cloud-qwen32b] Dataset: ${DATASET_PATH}"
echo "[cloud-qwen32b] Output: ${OUTPUT_DIR}"
nvidia-smi
df -h .

test -f "${DATASET_PATH}/train.jsonl"
test -f "${DATASET_PATH}/eval.jsonl"

python -m compileall -q scripts tests
python -m pytest tests/test_prepare_sft_dataset.py tests/test_evaluate_qlora_adapters.py tests/test_train_qlora.py -q

python -m scripts.train_qlora \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --train-rows "${TRAIN_ROWS}" \
  --eval-rows "${EVAL_ROWS}" \
  --max-steps "${MAX_STEPS}" \
  --max-seq-length "${SEQ_LENGTH}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-batches "${EVAL_BATCHES}" \
  --manual-loop \
  "${LOCAL_FLAG[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/cloud_train.log"

test -f "${OUTPUT_DIR}/train_metrics.json"
test -f "${OUTPUT_DIR}/final_adapter/adapter_config.json"

if [[ "${RUN_EVAL}" == "1" ]]; then
  python -m scripts.evaluate_qlora_adapters \
    --config "${CONFIG_PATH}" \
    --dataset auto \
    --out-dir "outputs/qlora_eval/qwen25_32b_cloud_validation_limit64" \
    --adapter-path "${OUTPUT_DIR}/final_adapter" \
    --adapter-name qwen32b_cleaned_sft_step100 \
    --limit 64 \
    --seed 42 \
    --max-new-tokens 32 \
    --max-seq-length "${SEQ_LENGTH}" \
    --local-files-only \
    --skip-base \
    2>&1 | tee "${OUTPUT_DIR}/cloud_eval_adapter.log"
fi

echo "[cloud-qwen32b] Completed. Output directory: ${OUTPUT_DIR}"
