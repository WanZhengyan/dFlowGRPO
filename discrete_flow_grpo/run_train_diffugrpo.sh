#!/bin/bash
# =============================================================================
# DiffuGRPO Training (per-sample geometric-mean ratio over D image-token dims;
# single forward of pi_theta at x_0; only x_0, x_1 and log p_old(x_1|x_0) saved
# during sampling — much cheaper memory than (G)SPO).
# =============================================================================
# Usage:
#   bash run_train_diffugrpo.sh                              # defaults: pickscore reward
#   REWARD='{"aesthetic": 1}' bash run_train_diffugrpo.sh    # aesthetic reward (auto → pickscore dataset)
#   REWARD='{"geneval": 1}' bash run_train_diffugrpo.sh      # geneval reward (auto → geneval dataset)
#   REWARD='{"ocr": 1}' bash run_train_diffugrpo.sh          # ocr reward (auto → ocr dataset)
#   DATASET=drawbench REWARD='{"clip": 1}' bash run_train_diffugrpo.sh  # manual dataset override
#   RUN_TAG=exp01 bash run_train_diffugrpo.sh                # custom run name
#   NUM_GPUS=4 bash run_train_diffugrpo.sh                   # 4 GPUs
#   RESUME_STEP=500 bash run_train_diffugrpo.sh              # resume from step 500
#   USE_CFG=1 bash run_train_diffugrpo.sh                    # enable CFG (appends _cfg)
#
# Naming convention:
#   Output dir: output_diffugrpo_<REWARD_TAG>[_<RUN_TAG>]/
#   Log file:   logs/diffugrpo_<REWARD_TAG>[_<RUN_TAG>].log
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
REWARD=${REWARD:-'{"pickscore": 1}'}            # reward function (JSON dict)
# DATASET auto-detection: geneval reward → geneval dataset, otherwise → pickscore
if [ -z "${DATASET:-}" ]; then
    if echo "$REWARD" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'geneval' in d else 1)"; then
        DATASET=geneval
    elif echo "$REWARD" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'ocr' in d else 1)"; then
        DATASET=ocr
    else
        DATASET=pickscore
    fi
fi
KL_BETA=${KL_BETA:-0}                           # KL penalty coefficient (0 = disabled)
RUN_TAG=${RUN_TAG:-token_clip}                  # custom suffix for output dir name
RESUME_STEP=${RESUME_STEP:-0}                   # 0 = train from scratch; N = resume from step N
USE_CFG=${USE_CFG:-0}                           # 0 = no CFG (--no_cfg); 1 = enable CFG (appends _cfg)
NUM_GPUS=${NUM_GPUS:-8}                         # number of GPUs
EXTRA_ARGS=${EXTRA_ARGS:-}                      # extra CLI flags for train_DiffuGRPO.py
SAMPLE_BATCH_SIZE=24
TRAIN_BATCH_SIZE=16
# =============================================================================

DATASET_DIR=/your_path/dFlowGRPO/flow_grpo/dataset/${DATASET}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ACC_CONFIG=${ACC_CONFIG:-${SCRIPT_DIR}/accelerate_config_ds.yaml}

# ---- Build output directory name from reward ----
REWARD_TAG=$(echo "$REWARD" | python3 -c "import sys,json; d=json.load(sys.stdin); print('_'.join(sorted(d.keys())))")
if [ -n "$RUN_TAG" ]; then
    OUTPUT_NAME="grpo_diffu_${REWARD_TAG}_${RUN_TAG}"
else
    OUTPUT_NAME="grpo_diffu_${REWARD_TAG}"
fi
# Append _cfg suffix when CFG is enabled
_NO_CFG_FLAG="--no_cfg"
if [ "${USE_CFG}" -eq 1 ] 2>/dev/null; then
    OUTPUT_NAME="${OUTPUT_NAME}_cfg"
    _NO_CFG_FLAG=""
fi
# Append _klX.XX suffix when KL penalty is enabled
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
        echo "[resume] WARNING: No checkpoint found for step ${RESUME_STEP} in ${OUTPUT_DIR}"
        echo "[resume]          (looked for checkpoint_*_step${RESUME_STEP})"
        echo "[resume]          Training from scratch."
    fi
fi

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/${OUTPUT_NAME}.log"
touch "${LOG_FILE}"

# ---- Auto-detect --reward_on_gpu ----
_GPU_REWARD_FLAG=""
if echo "$REWARD" | python3 -c "
import sys, json
d = json.load(sys.stdin)
cpu_only = {'geneval', 'jpeg', 'deqa', 'unifiedreward'}
if not set(d.keys()).issubset(cpu_only):
    sys.exit(0)
sys.exit(1)
"; then
    _GPU_REWARD_FLAG="--reward_on_gpu"
fi

# ---- Check geneval reward server (if geneval reward is used) ----
REWARD_SERVER_PORT=18085

_needs_geneval_server() {
    echo "$REWARD" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'geneval' in d else 1)"
}

if _needs_geneval_server; then
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${REWARD_SERVER_PORT}/" 2>/dev/null | grep -q '200\|405\|400'; then
        echo "[reward-server] Running on port ${REWARD_SERVER_PORT}. ✓"
    else
        echo "ERROR: GenEval reward server is NOT running on port ${REWARD_SERVER_PORT}."
        echo "Start it first:  tmux attach -t reward_server  (see note.txt)"
        exit 1
    fi
fi

echo "============================================="
echo "  DiffuGRPO Training"
echo "  Reward:     ${REWARD}"
echo "  Dataset:    ${DATASET_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Log:        ${LOG_FILE}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Reward GPU: ${_GPU_REWARD_FLAG:-off}"
echo "  CFG:        $([ -z "${_NO_CFG_FLAG}" ] && echo 'on' || echo 'off')"
echo "  KL beta:    ${KL_BETA}"
if [ -n "${_RESUME_FLAG}" ]; then
echo "  Resume:     step ${RESUME_STEP}"
else
echo "  Resume:     off (training from scratch)"
fi
echo "============================================="

accelerate launch \
    --config_file "${ACC_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    "${SCRIPT_DIR}/train_DiffuGRPO.py" \
    --checkpoint_path "${CKPT_PATH}" \
    --text_embedding_path "${CKPT_PATH}/text_embedding.pt" \
    --image_embedding_path "${CKPT_PATH}/image_embedding.pt" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --ema \
    --use_lora \
    ${_NO_CFG_FLAG} \
    --reward_dict "${REWARD}" \
    --global_std \
    --token_level_loss \
    --sample_batch_size "${SAMPLE_BATCH_SIZE}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --wandb_name "${OUTPUT_NAME}" \
    ${_GPU_REWARD_FLAG} \
    ${_KL_FLAG} \
    ${_RESUME_FLAG} \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"
