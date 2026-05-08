#!/bin/bash
# =============================================================================
# DrawBench Evaluation — score trained checkpoints on the DrawBench test set
# with one of six reward models:
#
#   aesthetic | deqa | imagereward | pickscore | unifiedreward | hpsv3
#
# Backed by evaluate_drawbench.py, which keeps a SHARED IMAGE CACHE under
# <base_out_dir>/_image_cache/<suffix>/  (manifest.json + images/*.png).
# The first reward you score for a given (checkpoint, suffix) generates the
# images; subsequent rewards reuse the cached PNGs and only run scoring.
#
# Output layout (UNCHANGED from before, with a sibling cache dir):
#   eval_drawbench_<train_reward>_results/
#     _image_cache/<suffix>/{manifest.json, images/*.png}     # NEW
#     <eval_reward>/<suffix>/results.json
#     <eval_reward>/summary.json                              # batch mode
#
# Usage examples:
#   # Single checkpoint, PickScore:
#   EVAL_REWARD=pickscore STEP=2600 BATCH=0 bash run_eval_drawbench.sh
#
#   # Batch mode over specific steps, HPSv3:
#   EVAL_REWARD=hpsv3 BATCH=1 EVAL_STEPS="1000,2000,2600" bash run_eval_drawbench.sh
#
#   # Batch mode at a fixed interval, Aesthetic:
#   EVAL_REWARD=aesthetic BATCH=1 STEP_INTERVAL=500 EVAL_STEPS="" bash run_eval_drawbench.sh
#
#   # Pre-build the image cache once (no reward), then score later:
#   EVAL_REWARD=aesthetic IMAGE_CACHE_ONLY=1 BATCH=1 \
#       EVAL_STEPS="1000,2000,3200" bash run_eval_drawbench.sh
#
# Notes:
#   * `deqa` and `unifiedreward` require their reward servers to be running
#     (see flow_grpo/ rewards.py and DiscreteFlowRL/reward-server/).
#   * `hpsv3` requires `pip install hpsv3` (weights auto-download on first use).
# =============================================================================

set -euo pipefail

source /home/your_path/miniconda3/etc/profile.d/conda.sh
conda activate dflow_grpo

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/your_path/dFlowGRPO/FUDOKI:/home/your_path/dFlowGRPO/flow_grpo:/home/your_path/dFlowGRPO/discrete_flow_grpo:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}

# =============================================================================
# >>>>>>>>>> CUSTOMIZE HERE <<<<<<<<<<
# =============================================================================

# ---- Paths ----
CKPT_PATH=${CKPT_PATH:-/home/your_path/dFlowGRPO/FUDOKI/checkpoints}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ACC_CONFIG=${ACC_CONFIG:-${SCRIPT_DIR}/accelerate_config.yaml}

# ---- Training run to evaluate ----
TRAIN_REWARD=${TRAIN_REWARD:-pickscore}
GRPO_DIR=${GRPO_DIR:-${SCRIPT_DIR}/output_grpo_${TRAIN_REWARD}}

# ---- Evaluation reward(s) ----
#   Allowed: aesthetic | deqa | imagereward | pickscore | unifiedreward | hpsv3
#
#   * EVAL_REWARDS : space-separated list, e.g. "aesthetic pickscore hpsv3"
#                    The script will loop over them; the first one generates
#                    the image cache, the rest reuse it (scoring only).
#   * EVAL_REWARD  : kept for backward compatibility (single reward).
EVAL_REWARDS=${EVAL_REWARDS:-${EVAL_REWARD:-"unifiedreward"}}

# Validate every entry up-front.
for _r in ${EVAL_REWARDS}; do
    case "${_r}" in
        aesthetic|deqa|imagereward|pickscore|unifiedreward|hpsv3)
            ;;
        *)
            echo "ERROR: EVAL_REWARDS entry must be one of: aesthetic, deqa, imagereward, pickscore, unifiedreward, hpsv3"
            echo "       Got: '${_r}'"
            exit 1
            ;;
    esac
done

# ---- DrawBench test prompts (fixed) ----
TEST_PROMPTS=${TEST_PROMPTS:-/your_path/dFlowGRPO/flow_grpo/dataset/drawbench/test.txt}

# ---- Mode: single vs batch ----
BATCH=${BATCH:-1}                  # 0 = single checkpoint; 1 = batch mode
STEP=${STEP:-}                     # Step number for single mode
STEP_INTERVAL=${STEP_INTERVAL:-}   # Evaluate every N steps in batch mode
EVAL_STEPS=${EVAL_STEPS:-"6300"}   # Comma-separated steps to eval (overrides STEP_INTERVAL)

# ---- Eval options ----
USE_CFG=${USE_CFG:-0}
EMA_ONLY=${EMA_ONLY:-1}
USE_EMA=${USE_EMA:-1}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SKIP_BASELINE=${SKIP_BASELINE:-0}
NUM_IMAGES=${NUM_IMAGES:-1}
NUM_GPUS=${NUM_GPUS:-7}
SEED=${SEED:-42}
REWARD_ON_GPU=${REWARD_ON_GPU:-1}
NUM_PROMPTS=${NUM_PROMPTS:-0}             # 0 = use all drawbench prompts
EVAL_FM_STEPS=${EVAL_FM_STEPS:-20}
IMAGE_CACHE_ONLY=${IMAGE_CACHE_ONLY:-0}   # 1 = generate cache only; skip scoring

# =============================================================================
# Derived flags / output dir
# =============================================================================
_CFG_SUFFIX=""
_NO_CFG_FLAG=""
if [ "${USE_CFG}" -eq 0 ] 2>/dev/null; then
    _NO_CFG_FLAG="--no_cfg"
else
    _CFG_SUFFIX="_cfg"
fi

_NFE_SUFFIX="_${EVAL_FM_STEPS}nfe"

# Output base dir (shared across all rewards in this run).
_BASE_OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/eval_drawbench_${TRAIN_REWARD}_results}
mkdir -p "${_BASE_OUT_DIR}/_image_cache"

echo "============================================="
echo "  DrawBench Evaluation (cache-aware, multi-reward)"
echo "  Train reward:  ${TRAIN_REWARD}"
echo "  Eval rewards:  ${EVAL_REWARDS}"
echo "  Base out dir:  ${_BASE_OUT_DIR}"
echo "============================================="

# =============================================================================
# Loop over every requested reward. The first one will populate the image
# cache; subsequent ones reuse the cached PNGs and only run scoring.
# =============================================================================
for EVAL_REWARD in ${EVAL_REWARDS}; do

REWARD="{\"${EVAL_REWARD}\": 1}"
OUT_DIR="${_BASE_OUT_DIR}/${EVAL_REWARD}"
mkdir -p "${OUT_DIR}/logs"

# Common flags shared by both modes
_COMMON_FLAGS=""
[ "${REWARD_ON_GPU}"     -eq 1 ] 2>/dev/null && _COMMON_FLAGS="${_COMMON_FLAGS} --reward_on_gpu"
[ "${IMAGE_CACHE_ONLY}"  -eq 1 ] 2>/dev/null && _COMMON_FLAGS="${_COMMON_FLAGS} --image_cache_only"
[ -n "${_NO_CFG_FLAG}" ] && _COMMON_FLAGS="${_COMMON_FLAGS} ${_NO_CFG_FLAG}"

# =============================================================================
# Print configuration
# =============================================================================
echo "============================================="
echo "  DrawBench Evaluation (cache-aware)"
echo "  Train reward: ${TRAIN_REWARD}"
echo "  Eval reward:  ${EVAL_REWARD}"
echo "  Test data:    ${TEST_PROMPTS}"
echo "  GRPO dir:     ${GRPO_DIR}"
echo "  Output dir:   ${OUT_DIR}"
echo "  Cache dir:    ${_BASE_OUT_DIR}/_image_cache"
echo "  Mode:         $([ "${BATCH}" -eq 1 ] && echo 'BATCH' || echo 'SINGLE')"
if [ "${BATCH}" -eq 1 ]; then
    if [ -n "${EVAL_STEPS}" ]; then
        echo "  Steps:        ${EVAL_STEPS}"
    else
        echo "  Interval:     every ${STEP_INTERVAL} steps"
    fi
    echo "  EMA only:     $([ "${EMA_ONLY}" -eq 1 ] && echo 'yes' || echo 'no')"
    echo "  Skip exist:   $([ "${SKIP_EXISTING}" -eq 1 ] && echo 'yes' || echo 'no')"
    echo "  Skip base:    $([ "${SKIP_BASELINE}" -eq 1 ] && echo 'yes' || echo 'no')"
else
    echo "  Step:         ${STEP}"
    echo "  EMA:          $([ "${USE_EMA}" -eq 1 ] && echo 'yes' || echo 'no')"
fi
echo "  CFG:          $([ "${USE_CFG}" -eq 1 ] && echo 'on' || echo 'off')"
echo "  Cache only:   $([ "${IMAGE_CACHE_ONLY}" -eq 1 ] && echo 'yes' || echo 'no')"
echo "  GPUs:         ${NUM_GPUS}"
echo "  NFE:          ${EVAL_FM_STEPS}"
echo "  Imgs/prompt:  ${NUM_IMAGES}"
echo "  Seed:         ${SEED}"
echo "============================================="

# =============================================================================
# Run
# =============================================================================
if [ "${BATCH}" -eq 1 ]; then
    extra_flags="${_COMMON_FLAGS}"
    [ "${EMA_ONLY}"      -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --ema_only"
    [ "${SKIP_EXISTING}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --skip_existing"
    [ "${SKIP_BASELINE}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --skip_baseline"

    step_arg=""
    if [ -n "${EVAL_STEPS}" ]; then
        step_arg="--eval_steps ${EVAL_STEPS}"
    elif [ -n "${STEP_INTERVAL}" ]; then
        step_arg="--step_interval ${STEP_INTERVAL}"
    else
        echo "ERROR: In BATCH mode, set EVAL_STEPS or STEP_INTERVAL."
        exit 1
    fi

    log_name="eval_drawbench_${EVAL_REWARD}${_NFE_SUFFIX}${_CFG_SUFFIX}"
    echo ""
    echo "===== DrawBench BATCH (${EVAL_REWARD}) on ${NUM_GPUS} GPUs ====="
    accelerate launch \
        --config_file "${ACC_CONFIG}" \
        --num_processes "${NUM_GPUS}" \
        "${SCRIPT_DIR}/evaluate_drawbench.py" \
        --checkpoint_path "${CKPT_PATH}" \
        --image_embedding_path "${CKPT_PATH}/image_embedding.pt" \
        --test_prompts_file "${TEST_PROMPTS}" \
        --grpo_output_dir "${GRPO_DIR}" \
        --output_dir "${OUT_DIR}" \
        --reward_dict "${REWARD}" \
        --num_prompts "${NUM_PROMPTS}" \
        --num_images_per_prompt "${NUM_IMAGES}" \
        --discrete_fm_steps "${EVAL_FM_STEPS}" \
        --seed "${SEED}" \
        ${extra_flags} \
        ${step_arg} \
        2>&1 | tee "${OUT_DIR}/logs/${log_name}.log"
    echo "===== DrawBench BATCH (${EVAL_REWARD}) complete ====="
else
    # ---- SINGLE MODE ----
    if [ -z "${STEP}" ]; then
        echo "ERROR: Set STEP=<n> in single mode."
        exit 1
    fi
    _ckpt_dir=$(find "${GRPO_DIR}" -maxdepth 1 -type d -name "checkpoint_*_step${STEP}" 2>/dev/null | head -1)
    if [ -z "${_ckpt_dir}" ]; then
        echo "ERROR: No checkpoint found for step ${STEP} in ${GRPO_DIR}"
        ls -d "${GRPO_DIR}"/checkpoint_*_step* 2>/dev/null | head -20 || echo "  (none)"
        exit 1
    fi

    if [ "${USE_EMA}" -eq 1 ] && [ -d "${_ckpt_dir}/lora_adapter_ema" ]; then
        LORA_PATH="${_ckpt_dir}/lora_adapter_ema"
        _suffix="step${STEP}_ema${_NFE_SUFFIX}${_CFG_SUFFIX}"
    elif [ -d "${_ckpt_dir}/lora_adapter" ]; then
        LORA_PATH="${_ckpt_dir}/lora_adapter"
        _suffix="step${STEP}${_NFE_SUFFIX}${_CFG_SUFFIX}"
    else
        echo "ERROR: No lora_adapter found in ${_ckpt_dir}"
        exit 1
    fi

    # Skip if results.json already exists for this reward+suffix.
    if [ "${SKIP_EXISTING}" -eq 1 ] 2>/dev/null && \
       [ "${IMAGE_CACHE_ONLY}" -eq 0 ] 2>/dev/null && \
       [ -f "${OUT_DIR}/${_suffix}/results.json" ]; then
        echo "[SKIP] ${_suffix} (results exist)"
        exit 0
    fi

    extra_flags="${_COMMON_FLAGS}"
    log_name="eval_${_suffix}_${EVAL_REWARD}"

    echo "Using LoRA: ${LORA_PATH}"
    echo ""
    echo "===== DrawBench SINGLE step${STEP}${_CFG_SUFFIX} (${EVAL_REWARD}) on ${NUM_GPUS} GPUs ====="
    accelerate launch \
        --config_file "${ACC_CONFIG}" \
        --num_processes "${NUM_GPUS}" \
        "${SCRIPT_DIR}/evaluate_drawbench.py" \
        --checkpoint_path "${CKPT_PATH}" \
        --image_embedding_path "${CKPT_PATH}/image_embedding.pt" \
        --test_prompts_file "${TEST_PROMPTS}" \
        --lora_path "${LORA_PATH}" \
        --output_dir "${OUT_DIR}/${_suffix}" \
        --reward_dict "${REWARD}" \
        --num_prompts "${NUM_PROMPTS}" \
        --num_images_per_prompt "${NUM_IMAGES}" \
        --discrete_fm_steps "${EVAL_FM_STEPS}" \
        --seed "${SEED}" \
        ${extra_flags} \
        2>&1 | tee "${OUT_DIR}/logs/${log_name}.log"
    echo "===== DrawBench SINGLE step${STEP}${_CFG_SUFFIX} (${EVAL_REWARD}) complete ====="
fi

done   # ---- end of EVAL_REWARDS loop ----

echo ""
echo "All DrawBench evaluations complete."
echo "Results root:        ${_BASE_OUT_DIR}"
echo "Per-reward subdirs:  ${EVAL_REWARDS}"
echo "Shared image cache:  ${_BASE_OUT_DIR}/_image_cache"
