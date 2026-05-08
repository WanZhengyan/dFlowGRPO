#!/bin/bash
# =============================================================================
# GRPO Training — Multimodal Understanding (ScienceQA MCQ)
# =============================================================================
# Usage:
#   bash run_train_grpo_understanding.sh                # defaults
#   RUN_TAG=exp01 bash run_train_grpo_understanding.sh  # custom run name
#   NUM_GPUS=4 bash run_train_grpo_understanding.sh     # 4 GPUs
#   RESUME_STEP=500 bash run_train_grpo_understanding.sh
#
# Naming convention:
#   Output dir: output_grpo_u_scienceqa[_<RUN_TAG>]/
#   Log file:   logs/grpo_u_scienceqa[_<RUN_TAG>].log
# =============================================================================

set -euo pipefail

# ---- Environment ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
source /your_path/miniconda3/etc/profile.d/conda.sh
conda activate dflow_grpo

export PYTHONNOUSERSITE=1
export PYTHONPATH=/your_path/dFlowGRPO/FUDOKI:/your_path/dFlowGRPO/flow_grpo:/your_path/dFlowGRPO/discrete_flow_grpo:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# >>>>>>>>>> CUSTOMIZE HERE <<<<<<<<<<
# =============================================================================
CKPT_PATH=${CKPT_PATH:-/your_path/dFlowGRPO/FUDOKI/checkpoints}
DATASET=${DATASET:-ScienceQA}
KL_BETA=${KL_BETA:-0.01}
RUN_TAG=${RUN_TAG:-}
RESUME_STEP=${RESUME_STEP:-}
NUM_GPUS=${NUM_GPUS:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-24}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
TXT_MAX_LENGTH=${TXT_MAX_LENGTH:-500}

# ---- API judge (optional LLM-assisted answer extraction) ----
# Set USE_API_JUDGE=1 to enable. Fill in OPENAI_API_KEY (and optionally
# API_JUDGE_BASE_URL for DeepSeek / local vLLM / etc.).
USE_API_JUDGE=${USE_API_JUDGE:-1}
OPENAI_API_KEY=${OPENAI_API_KEY:-your_openai_api_key_here}            # <-- put your key here (or export it before running)
API_JUDGE_MODEL=${API_JUDGE_MODEL:-gpt-4o-mini}
API_JUDGE_BASE_URL=${API_JUDGE_BASE_URL:-}    # e.g. https://api.deepseek.com/v1
API_JUDGE_MAX_WORKERS=${API_JUDGE_MAX_WORKERS:-8}
API_JUDGE_ONLY_EVAL=${API_JUDGE_ONLY_EVAL:-0} # 1 = judge only at eval time (save API budget)
# =============================================================================

# Export key so the Python side can read $OPENAI_API_KEY.
if [ -n "${OPENAI_API_KEY}" ]; then
    export OPENAI_API_KEY
fi

_API_JUDGE_FLAGS=""
if [ "${USE_API_JUDGE}" = "1" ] || [ "${USE_API_JUDGE}" = "true" ]; then
    if [ -z "${OPENAI_API_KEY}" ]; then
        echo "[api_judge] ERROR: USE_API_JUDGE=1 but OPENAI_API_KEY is empty." >&2
        exit 1
    fi
    _API_JUDGE_FLAGS="--use_api_judge --api_judge_model ${API_JUDGE_MODEL} --api_judge_max_workers ${API_JUDGE_MAX_WORKERS}"
    if [ -n "${API_JUDGE_BASE_URL}" ]; then
        _API_JUDGE_FLAGS="${_API_JUDGE_FLAGS} --api_judge_base_url ${API_JUDGE_BASE_URL}"
    fi
    if [ "${API_JUDGE_ONLY_EVAL}" = "1" ] || [ "${API_JUDGE_ONLY_EVAL}" = "true" ]; then
        _API_JUDGE_FLAGS="${_API_JUDGE_FLAGS} --api_judge_only_eval"
    fi
fi

DATASET_DIR=/your_path/dFlowGRPO/discrete_flow_grpo/dataset_understanding/${DATASET}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ACC_CONFIG=${ACC_CONFIG:-${SCRIPT_DIR}/accelerate_config_ds.yaml}

# ---- Build output dir name ----
DATASET_TAG=$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')
if [ -n "$RUN_TAG" ]; then
    OUTPUT_NAME="grpo_u_${DATASET_TAG}_${RUN_TAG}"
else
    OUTPUT_NAME="grpo_u_${DATASET_TAG}"
fi
# Append _klX.XX suffix if KL enabled
_KL_FLAG=""
if python3 -c "import sys; sys.exit(0 if float('${KL_BETA}') > 0 else 1)" 2>/dev/null; then
    OUTPUT_NAME="${OUTPUT_NAME}_kl${KL_BETA}"
    _KL_FLAG="--kl_beta ${KL_BETA}"
fi
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/output_${OUTPUT_NAME}}

# ---- Auto-detect resume checkpoint ----
_RESUME_FLAG=""
if [ "${RESUME_STEP}" -gt 0 ] 2>/dev/null; then
    _RESUME_CKPT=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name "checkpoint_*_step${RESUME_STEP}" 2>/dev/null | head -1)
    if [ -n "${_RESUME_CKPT}" ]; then
        echo "[resume] Found checkpoint at: ${_RESUME_CKPT}"
        _RESUME_FLAG="--resume_from_checkpoint ${_RESUME_CKPT}"
    else
        echo "[resume] WARNING: No checkpoint for step ${RESUME_STEP} in ${OUTPUT_DIR}. Training from scratch."
    fi
fi

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/${OUTPUT_NAME}.log"
touch "${LOG_FILE}"

echo "============================================="
echo "  GRPO Understanding Training"
echo "  Dataset:    ${DATASET_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Log:        ${LOG_FILE}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  KL beta:    ${KL_BETA}"
echo "  Sample bs:  ${SAMPLE_BATCH_SIZE}"
echo "  Train bs:   ${TRAIN_BATCH_SIZE}"
if [ -n "${_API_JUDGE_FLAGS}" ]; then
echo "  API judge:  on  (model=${API_JUDGE_MODEL}, only_eval=${API_JUDGE_ONLY_EVAL})"
else
echo "  API judge:  off"
fi
if [ -n "${_RESUME_FLAG}" ]; then
echo "  Resume:     step ${RESUME_STEP}"
else
echo "  Resume:     off (training from scratch)"
fi
echo "============================================="

accelerate launch \
    --config_file "${ACC_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    "${SCRIPT_DIR}/train_GRPO_understanding.py" \
    --checkpoint_path "${CKPT_PATH}" \
    --text_embedding_path "${CKPT_PATH}/text_embedding.pt" \
    --image_embedding_path "${CKPT_PATH}/image_embedding.pt" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --ema \
    --use_lora \
    --global_std \
    --sample_batch_size "${SAMPLE_BATCH_SIZE}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --txt_max_length "${TXT_MAX_LENGTH}" \
    --wandb_name "${OUTPUT_NAME}" \
    ${_KL_FLAG} \
    ${_RESUME_FLAG} \
    ${_API_JUDGE_FLAGS} \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"
