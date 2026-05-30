#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config/qlora_cloud_qwen25_coder_32b.yml}"
DATASET_PATH="${DATASET_PATH:-outputs/sft_dataset/qwen_cyber_sft_v2_20k}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qlora/qwen25_coder_32b_cloud_validation_r16_512_step100}"
TRAIN_ROWS="${TRAIN_ROWS:-4096}"
EVAL_ROWS="${EVAL_ROWS:-512}"
MAX_STEPS="${MAX_STEPS:-100}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-3}"
EVAL_EVERY="${EVAL_EVERY:-25}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/qlora_eval/qwen25_32b_cloud_validation_limit64}"
EVAL_LIMIT="${EVAL_LIMIT:-64}"
ADAPTER_NAME="${ADAPTER_NAME:-qwen32b_cleaned_sft_step100}"
RUN_NAME="${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
REPORT_DIR="${REPORT_DIR:-reports/qwen32b_runs/${RUN_NAME}}"
SYNC_GIT_ARTIFACTS="${SYNC_GIT_ARTIFACTS:-1}"
GIT_PUSH_ARTIFACTS="${GIT_PUSH_ARTIFACTS:-1}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  LOCAL_FLAG=(--local-files-only)
else
  LOCAL_FLAG=()
fi

SYNC_FLAGS=()
if [[ "${SYNC_GIT_ARTIFACTS}" == "1" ]]; then
  SYNC_FLAGS=(--sync-git-artifacts --artifact-report-dir "${REPORT_DIR}" --artifact-run-name "${RUN_NAME}")
  if [[ "${GIT_PUSH_ARTIFACTS}" != "1" ]]; then
    SYNC_FLAGS+=(--no-git-push-artifacts)
  fi
else
  SYNC_FLAGS=(--no-sync-git-artifacts)
fi

sync_run_artifacts() {
  if [[ "${SYNC_GIT_ARTIFACTS}" != "1" ]]; then
    return 0
  fi
  PUSH_FLAG=()
  if [[ "${GIT_PUSH_ARTIFACTS}" != "1" ]]; then
    PUSH_FLAG=(--no-push)
  fi
  python -m scripts.sync_run_artifacts \
    --run-dir "${OUTPUT_DIR}" \
    --eval-dir "${EVAL_OUTPUT_DIR}" \
    --report-dir "${REPORT_DIR}" \
    --run-name "${RUN_NAME}" \
    --remote "${GIT_REMOTE}" \
    --branch "${GIT_BRANCH}" \
    "${PUSH_FLAG[@]}" || true
}

trap sync_run_artifacts EXIT

mkdir -p "${OUTPUT_DIR}"

echo "[cloud-qwen32b] Python: $(python --version)"
echo "[cloud-qwen32b] Config: ${CONFIG_PATH}"
echo "[cloud-qwen32b] Dataset: ${DATASET_PATH}"
echo "[cloud-qwen32b] Output: ${OUTPUT_DIR}"
echo "[cloud-qwen32b] Report artifacts: ${REPORT_DIR}"
echo "[cloud-qwen32b] HF_HOME: ${HF_HOME}"
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
  --keep-checkpoints "${KEEP_CHECKPOINTS}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-batches "${EVAL_BATCHES}" \
  --resume \
  --manual-loop \
  "${SYNC_FLAGS[@]}" \
  "${LOCAL_FLAG[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/cloud_train.log"

test -f "${OUTPUT_DIR}/train_metrics.json"
test -f "${OUTPUT_DIR}/final_adapter/adapter_config.json"

if [[ "${RUN_EVAL}" == "1" ]]; then
  python -m scripts.evaluate_qlora_adapters \
    --config "${CONFIG_PATH}" \
    --dataset auto \
    --out-dir "${EVAL_OUTPUT_DIR}" \
    --adapter-path "${OUTPUT_DIR}/final_adapter" \
    --adapter-name "${ADAPTER_NAME}" \
    --limit "${EVAL_LIMIT}" \
    --seed 42 \
    --max-new-tokens 32 \
    --max-seq-length "${SEQ_LENGTH}" \
    --local-files-only \
    --skip-base \
    2>&1 | tee "${OUTPUT_DIR}/cloud_eval_adapter.log"
fi

sync_run_artifacts

echo "[cloud-qwen32b] Completed. Output directory: ${OUTPUT_DIR}"
