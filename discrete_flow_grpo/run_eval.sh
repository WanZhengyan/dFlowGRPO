#!/bin/bash
# =============================================================================
# GenEval / PickScore Evaluation — flexible multi-GPU evaluation script.
#
# Only three reward modes are supported here (use run_eval_drawbench.sh for
# the wider DrawBench reward zoo):
#   - EVAL_REWARD=geneval   → evaluate_geneval.py    (needs reward server on :18085)
#   - EVAL_REWARD=pickscore → evaluate_checkpoint.py (PickScore in-process)
#   - EVAL_REWARD=ocr       → evaluate_checkpoint.py (PaddleOCR in-process,
#                                                    uses dataset/ocr/test.txt)
#
# Modes:
#   1. Single checkpoint    (BATCH=0, STEP=100)
#   2. Multi-checkpoint     (BATCH=1, STEP_INTERVAL=100)
#   3. Range of steps       (BATCH=1, STEP_INTERVAL="100:200[:3000]")
#
# Other knobs: USE_CFG, EMA_ONLY, SKIP_EXISTING, SKIP_BASELINE.
# =============================================================================

set -euo pipefail

source /your_path/miniconda3/etc/profile.d/conda.sh
conda activate dflow_grpo

export PYTHONNOUSERSITE=1
export PYTHONPATH=/your_path/dFlowGRPO/FUDOKI:/your_path/dFlowGRPO/flow_grpo:/your_path/dFlowGRPO/discrete_flow_grpo:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# =============================================================================
# >>>>>>>>>> CUSTOMIZE HERE <<<<<<<<<<
# =============================================================================

# ---- Paths ----
CKPT_PATH=${CKPT_PATH:-/your_path/dFlowGRPO/FUDOKI/checkpoints}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ACC_CONFIG=${ACC_CONFIG:-${SCRIPT_DIR}/accelerate_config.yaml}

# ---- Training run to evaluate ----
# TRAIN_REWARD: the reward used during training (to locate checkpoint dir)
# EVAL_REWARD:  the reward/method used for evaluation. Must be one of
#               {pickscore, geneval}; for any other reward use
#               run_eval_drawbench.sh instead.
TRAIN_REWARD=${TRAIN_REWARD:-geneval}
EVAL_REWARD=${EVAL_REWARD:-geneval}

case "${EVAL_REWARD}" in
    geneval|pickscore|ocr) ;;
    *)
        echo "ERROR: run_eval.sh only supports EVAL_REWARD=geneval, pickscore or ocr."
        echo "       Got '${EVAL_REWARD}'. Use run_eval_drawbench.sh for other rewards."
        exit 1
        ;;
esac

GRPO_DIR=${GRPO_DIR:-${SCRIPT_DIR}/output_grpo_${TRAIN_REWARD}}

# ---- Reward dict for evaluation ----
REWARD="{\"${EVAL_REWARD}\": 1}"

# ---- Eval script selection (auto from EVAL_REWARD) ----
if [ "${EVAL_REWARD}" = "geneval" ]; then
    EVAL_MODE=geneval
else
    EVAL_MODE=checkpoint
fi

# ---- Mode: single vs batch ----
BATCH=${BATCH:-0}                    # 0 = single checkpoint; 1 = batch mode
STEP=${STEP:-3300}                       # Step number for single mode
# STEP_INTERVAL — controls which checkpoints get evaluated in batch mode.
#   - plain integer:           every N steps, e.g. STEP_INTERVAL=200
#   - range "start:step[:end]" e.g. "100:200"      (start=100, every 200, no upper bound)
#                                   "100:200:3100" (start=100, every 200, end=3100)
#                                   ":200"         (start=0,   every 200, no upper bound)
# Semantics: keeps existing checkpoint_*_step<N> dirs whose
#   step >= start, step <= end, (step - start) % stride == 0.
# Missing intermediate checkpoints are silently skipped.
STEP_INTERVAL=${STEP_INTERVAL:-"100:200"}

# ---- Eval options ----
USE_CFG=${USE_CFG:-0}               # 1 = enable CFG; 0 = no CFG
EMA_ONLY=${EMA_ONLY:-1}             # 1 = only evaluate EMA variants; 0 = both LoRA and LoRA+EMA
USE_EMA=${USE_EMA:-1}               # 1 = use EMA adapter; 0 = use non-EMA adapter (single mode)
SKIP_EXISTING=${SKIP_EXISTING:-1}   # 1 = skip already-evaluated checkpoints (batch mode)
SKIP_BASELINE=${SKIP_BASELINE:-0}   # 1 = skip baseline evaluation (batch mode)
NUM_IMAGES=${NUM_IMAGES:-1}         # Images per prompt
NUM_GPUS=${NUM_GPUS:-8}
SEED=${SEED:-42}
REWARD_ON_GPU=${REWARD_ON_GPU:-1}   # For non-geneval rewards: run scorer on GPU

# ---- GenEval-specific ----
# GenEval uses its own structured prompt set (test_metadata.jsonl).
GENEVAL_META=${GENEVAL_META:-/your_path/dFlowGRPO/flow_grpo/dataset/geneval/test_metadata.jsonl}

# ---- PickScore-specific ----
# PickScore uses a flat list of prompts (test.txt) — different from GenEval.
# OCR also uses a flat test.txt list, just from dataset/ocr/.
DATASET_DIR=/your_path/dFlowGRPO/flow_grpo/dataset
if [ "${EVAL_REWARD}" = "ocr" ]; then
    TEST_PROMPTS=${TEST_PROMPTS:-${DATASET_DIR}/ocr/test.txt}
else
    TEST_PROMPTS=${TEST_PROMPTS:-${DATASET_DIR}/pickscore/test.txt}
fi
NUM_PROMPTS=${NUM_PROMPTS:-0}       # 0 = use all prompts from test file

# ---- Shared sampling knob ----
EVAL_FM_STEPS=${EVAL_FM_STEPS:-20}  # Denoising NFE; default 20 for both pickscore and geneval

# =============================================================================

# ---- Build CFG flag ----
_CFG_SUFFIX=""
_NO_CFG_FLAG=""
if [ "${USE_CFG}" -eq 0 ] 2>/dev/null; then
    _NO_CFG_FLAG="--no_cfg"
else
    _CFG_SUFFIX="_cfg"
fi

# ---- Build NFE suffix (denoising steps) ----
_NFE_SUFFIX="_${EVAL_FM_STEPS}nfe"

# ---- Build output dir: eval_<train>_results/<eval_reward>/step... ----
# e.g. eval_pickscore_results/geneval/step3700_ema/
_eval_reward_name=$(echo "$REWARD" | python3 -c "import sys,json; d=json.load(sys.stdin); print('_'.join(d.keys()))" 2>/dev/null || echo "unknown")
_BASE_OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/eval_${TRAIN_REWARD}_results}
OUT_DIR="${_BASE_OUT_DIR}/${_eval_reward_name}"
mkdir -p "${OUT_DIR}/logs"

# ---- Parse STEP_INTERVAL as either a plain int or "start:step[:end]" ----
# Resulting variables passed to the underlying eval scripts:
#   _SI_STRIDE  — required, the modulus
#   _SI_START   — optional, lower bound (inclusive)
#   _SI_END     — optional, upper bound (inclusive)
_SI_STRIDE=""; _SI_START=""; _SI_END=""
if [ -n "${STEP_INTERVAL}" ]; then
    if [[ "${STEP_INTERVAL}" == *:* ]]; then
        _SI_START=$(echo "${STEP_INTERVAL}" | cut -d: -f1)
        _SI_STRIDE=$(echo "${STEP_INTERVAL}" | cut -d: -f2)
        _SI_END=$(echo "${STEP_INTERVAL}"   | cut -d: -f3)
        [ -z "${_SI_START}" ] && _SI_START=0
        if [ -z "${_SI_STRIDE}" ] || ! [ "${_SI_STRIDE}" -gt 0 ] 2>/dev/null; then
            echo "ERROR: STEP_INTERVAL='${STEP_INTERVAL}' has invalid stride '${_SI_STRIDE}'"
            exit 1
        fi
    else
        _SI_STRIDE="${STEP_INTERVAL}"
    fi
fi

# =============================================================================
# Helper functions
# =============================================================================

# Run GenEval evaluation on a single LoRA path
run_geneval_single() {
    local lora_path="$1"
    local out_subdir="$2"
    local log_name="$3"
    local tag_name="${4:-${out_subdir}}"

    local extra_flags=""
    [ -n "${_NO_CFG_FLAG}" ] && extra_flags="${extra_flags} ${_NO_CFG_FLAG}"

    echo ""
    echo "===== Evaluating: ${tag_name} on ${NUM_GPUS} GPUs ====="
    accelerate launch \
        --config_file "${ACC_CONFIG}" \
        --num_processes "${NUM_GPUS}" \
        "${SCRIPT_DIR}/evaluate_geneval.py" \
        --checkpoint_path "${CKPT_PATH}" \
        --image_embedding_path "${CKPT_PATH}/image_embedding.pt" \
        --geneval_metadata "${GENEVAL_META}" \
        --lora_path "${lora_path}" \
        --output_dir "${OUT_DIR}/${out_subdir}" \
        --num_images_per_prompt "${NUM_IMAGES}" \
        --discrete_fm_steps "${EVAL_FM_STEPS}" \
        --seed "${SEED}" \
        ${extra_flags} \
        2>&1 | tee "${OUT_DIR}/logs/${log_name}.log"
    echo "===== ${tag_name} evaluation complete ====="
}

# Run checkpoint (non-geneval) evaluation on a single LoRA path (multi-GPU)
run_checkpoint_single() {
    local lora_path="$1"
    local out_subdir="$2"
    local log_name="$3"
    local tag_name="${4:-${out_subdir}}"

    local extra_flags=""
    [ "${REWARD_ON_GPU}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --reward_on_gpu"
    [ -n "${_NO_CFG_FLAG}" ] && extra_flags="${extra_flags} ${_NO_CFG_FLAG}"

    echo ""
    echo "===== Evaluating: ${tag_name} on ${NUM_GPUS} GPUs ====="
    accelerate launch \
        --config_file "${ACC_CONFIG}" \
        --num_processes "${NUM_GPUS}" \
        "${SCRIPT_DIR}/evaluate_checkpoint.py" \
        --checkpoint_path "${CKPT_PATH}" \
        --image_embedding_path "${CKPT_PATH}/image_embedding.pt" \
        --test_prompts_file "${TEST_PROMPTS}" \
        --lora_path "${lora_path}" \
        --output_dir "${OUT_DIR}/${out_subdir}" \
        --reward_dict "${REWARD}" \
        --num_prompts "${NUM_PROMPTS}" \
        --num_images_per_prompt "${NUM_IMAGES}" \
        --discrete_fm_steps "${EVAL_FM_STEPS}" \
        --seed "${SEED}" \
        ${extra_flags} \
        2>&1 | tee "${OUT_DIR}/logs/${log_name}.log"
    echo "===== ${tag_name} evaluation complete ====="
}

# Run GenEval batch mode (built-in --grpo_output_dir)
run_geneval_batch() {
    local extra_flags=""
    [ -n "${_NO_CFG_FLAG}" ] && extra_flags="${extra_flags} ${_NO_CFG_FLAG}"
    [ "${EMA_ONLY}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --ema_only"
    [ "${SKIP_EXISTING}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --skip_existing"
    [ "${SKIP_BASELINE}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --skip_baseline"

    local step_arg=""
    [ -n "${_SI_STRIDE}" ] && step_arg="${step_arg} --step_interval ${_SI_STRIDE}"
    [ -n "${_SI_START}"  ] && step_arg="${step_arg} --step_start ${_SI_START}"
    [ -n "${_SI_END}"    ] && step_arg="${step_arg} --step_end ${_SI_END}"

    echo ""
    echo "===== GenEval BATCH evaluation on ${NUM_GPUS} GPUs ====="
    accelerate launch \
        --config_file "${ACC_CONFIG}" \
        --num_processes "${NUM_GPUS}" \
        "${SCRIPT_DIR}/evaluate_geneval.py" \
        --checkpoint_path "${CKPT_PATH}" \
        --image_embedding_path "${CKPT_PATH}/image_embedding.pt" \
        --geneval_metadata "${GENEVAL_META}" \
        --grpo_output_dir "${GRPO_DIR}" \
        --output_dir "${OUT_DIR}" \
        --num_images_per_prompt "${NUM_IMAGES}" \
        --discrete_fm_steps "${EVAL_FM_STEPS}" \
        --seed "${SEED}" \
        ${extra_flags} \
        ${step_arg} \
        2>&1 | tee "${OUT_DIR}/logs/eval_batch.log"
    echo "===== GenEval BATCH evaluation complete ====="
}

# Run checkpoint (non-geneval) batch mode (multi-GPU, built-in --grpo_output_dir)
run_checkpoint_batch() {
    local extra_flags=""
    [ "${REWARD_ON_GPU}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --reward_on_gpu"
    [ "${EMA_ONLY}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --ema_only"
    [ "${SKIP_EXISTING}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --skip_existing"
    [ "${SKIP_BASELINE}" -eq 1 ] 2>/dev/null && extra_flags="${extra_flags} --skip_baseline"
    [ -n "${_NO_CFG_FLAG}" ] && extra_flags="${extra_flags} ${_NO_CFG_FLAG}"

    local step_arg=""
    [ -n "${_SI_STRIDE}" ] && step_arg="${step_arg} --step_interval ${_SI_STRIDE}"
    [ -n "${_SI_START}"  ] && step_arg="${step_arg} --step_start ${_SI_START}"
    [ -n "${_SI_END}"    ] && step_arg="${step_arg} --step_end ${_SI_END}"

    echo ""
    echo "===== Checkpoint BATCH evaluation on ${NUM_GPUS} GPUs ====="
    accelerate launch \
        --config_file "${ACC_CONFIG}" \
        --num_processes "${NUM_GPUS}" \
        "${SCRIPT_DIR}/evaluate_checkpoint.py" \
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
        2>&1 | tee "${OUT_DIR}/logs/eval_batch.log"
    echo "===== Checkpoint BATCH evaluation complete ====="
}

# =============================================================================
# Print configuration
# =============================================================================
echo "============================================="
echo "  Evaluation Configuration"
echo "  Eval mode:   ${EVAL_MODE}"
echo "  Train reward:${TRAIN_REWARD}"
echo "  Eval reward: ${EVAL_REWARD} (${REWARD})"
if [ "${EVAL_MODE}" = "geneval" ]; then
    echo "  Test data:   ${GENEVAL_META}"
else
    echo "  Test data:   ${TEST_PROMPTS}"
fi
echo "  GRPO dir:    ${GRPO_DIR}"
echo "  Output dir:  ${OUT_DIR}"
echo "  Mode:        $([ "${BATCH}" -eq 1 ] && echo 'BATCH' || echo 'SINGLE')"
if [ "${BATCH}" -eq 1 ]; then
    if [ -n "${_SI_STRIDE}" ]; then
        if [ -n "${_SI_START}" ] || [ -n "${_SI_END}" ]; then
            echo "  Interval:    every ${_SI_STRIDE} steps (start=${_SI_START:-0}, end=${_SI_END:-max})"
        else
            echo "  Interval:    every ${_SI_STRIDE} steps"
        fi
    else
        echo "  Interval:    (all checkpoints)"
    fi
    echo "  EMA only:    $([ "${EMA_ONLY}" -eq 1 ] && echo 'yes' || echo 'no')"
    echo "  Skip exist:  $([ "${SKIP_EXISTING}" -eq 1 ] && echo 'yes' || echo 'no')"
    echo "  Skip base:   $([ "${SKIP_BASELINE}" -eq 1 ] && echo 'yes' || echo 'no')"
else
    echo "  Step:        ${STEP}"
    echo "  EMA:         $([ "${USE_EMA}" -eq 1 ] && echo 'yes' || echo 'no')"
fi
echo "  Denoise NFE: ${EVAL_FM_STEPS}"
echo "  CFG:         $([ "${USE_CFG}" -eq 1 ] && echo 'on' || echo 'off')"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Imgs/prompt: ${NUM_IMAGES}"
echo "  Seed:        ${SEED}"
echo "============================================="

# =============================================================================
# Run evaluation
# =============================================================================

if [ "${BATCH}" -eq 1 ]; then
    # ====================
    # BATCH MODE
    # ====================
    # Delegate to evaluate_geneval.py / evaluate_checkpoint.py with
    # --grpo_output_dir, which:
    #   - discovers checkpoint_*_step<N> dirs under GRPO_DIR
    #   - filters by --step_interval / --step_start / --step_end
    #   - honors --skip_existing (matches both legacy
    #       checkpoint_..._step<N>_ema/ and new step<N>_ema_<NFE>nfe[_cfg]/
    #       output dir layouts)
    #   - writes geneval_summary.json / summary.json with one entry per ckpt,
    #     including the baseline. This is what plots/plot_eval_curve.py reads.

    if [ "${EVAL_MODE}" = "geneval" ]; then
        run_geneval_batch
    else
        run_checkpoint_batch
    fi

else
    # ====================
    # SINGLE CHECKPOINT MODE
    # ====================

    # Find checkpoint directory for the given step
    _ckpt_dir=$(find -L "${GRPO_DIR}" -maxdepth 1 -type d -name "checkpoint_*_step${STEP}" 2>/dev/null | head -1)
    if [ -z "${_ckpt_dir}" ]; then
        echo "ERROR: No checkpoint found for step ${STEP} in ${GRPO_DIR}"
        echo "Available checkpoints:"
        ls -d "${GRPO_DIR}"/checkpoint_*_step* 2>/dev/null | head -20 || echo "  (none)"
        exit 1
    fi

    # Select LoRA adapter
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

    echo "Using LoRA: ${LORA_PATH}"

    if [ "${EVAL_MODE}" = "geneval" ]; then
        run_geneval_single "${LORA_PATH}" "${_suffix}" "eval_${_suffix}" "step${STEP}${_NFE_SUFFIX}${_CFG_SUFFIX}"
    else
        run_checkpoint_single "${LORA_PATH}" "${_suffix}" "eval_${_suffix}" "step${STEP}${_NFE_SUFFIX}${_CFG_SUFFIX}"
    fi
fi

echo ""
echo "All evaluations complete. Results in: ${OUT_DIR}"


