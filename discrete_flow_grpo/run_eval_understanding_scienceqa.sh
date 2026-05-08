#!/usr/bin/env bash
# filepath: /your_path/dFlowGRPO/discrete_flow_grpo/run_eval_understanding_scienceqa.sh
#
# Multi-GPU ScienceQA eval wrapper over eval_scienceqa.py (via accelerate).
#
# Three ways to pick the checkpoint(s):
#   (A) Auto-discover from a training run (like run_eval.sh):
#         TRAIN_REWARD=scienceqa_kl0.01   -> GRPO_DIR=output_grpo_u_<TRAIN_REWARD>
#         STEP=700                        -> finds  checkpoint_*_step700/
#         USE_EMA=1 (default)             -> output dir tagged _ema when
#                                            ema_state.pt exists next to
#                                            lora_adapter/ (auto-applied
#                                            by eval_scienceqa.py)
#   (B) Explicit path: LORA_PATH=/abs/path/to/lora_adapter
#   (C) BATCH mode (BATCH=1): iterate over many checkpoints in GRPO_DIR,
#         filtered by STEP_INTERVAL="start:stride[:end]" (mirrors run_eval.sh).
#           - "100:200"      -> start=100, every 200 steps, no upper bound
#           - "100:200:3100" -> start=100, every 200 steps, end=3100
#           - ":200"         -> start=0,   every 200 steps, no upper bound
#           - "200"          -> every 200 steps (start=0)
#         Set EMA_ONLY=1 (default) to only eval checkpoints with ema_state.pt
#         present (output dir tagged _ema). EMA_ONLY=0 also evaluates
#         checkpoints that only have lora_adapter/ (no EMA file).
#         Set SKIP_EXISTING=1 (default) to skip steps whose output dir exists.
#         Set SKIP_BASELINE=1 to skip the no-LoRA baseline run.
#
# If neither STEP nor LORA_PATH is provided (and BATCH=0) -> BASELINE (no LoRA).
#
# Inputs (env vars, all optional unless noted):
#   NUM_GPUS        number of GPUs                            (default 8)
#   CUDA_VISIBLE_DEVICES  GPU list                            (default 0..7)
#   CKPT_PATH       base FUDOKI checkpoint dir                (default: FUDOKI/checkpoints)
#   OPENAI_API_KEY  OpenAI key for the LLM answer extractor   (default: baked-in below)
#   TRAIN_REWARD    training tag -> GRPO_DIR                  (default scienceqa_kl0.01)
#   GRPO_DIR        explicit training-output dir              (default output_grpo_u_<TRAIN_REWARD>)
#   STEP            training step to eval (auto-discover EMA) (e.g. 700)
#   USE_EMA         1 = pick lora_adapter_ema, 0 = lora_adapter (default 1)
#   LORA_PATH       explicit LoRA adapter dir (overrides STEP)
#   EMA_PATH        EMA .pt (auto-detected next to LORA_PATH if unset)
#   GEN_MODULES_PATH  optional saved gen_modules .pt          (optional)
#   STEPS           denoising steps (top-1 per step)          (default 16)
#   SPLIT           test | val | train                        (default test)
#   NUM_SAMPLES     0 = all                                   (default 0)
#   API_MODEL       judge model name                          (default gpt-4o-mini)
#   API_BASE_URL    custom OpenAI-compat endpoint             (optional)
#   OUTPUT_DIR      where to dump predictions                 (optional)
#   ACC_CONFIG      accelerate config file                    (default accelerate_config_ds.yaml)
#
# Usage:
#   # Baseline (no LoRA)
#   bash run_eval_understanding_scienceqa.sh
#
#   # Auto-discover EMA for step 700 of output_grpo_u_scienceqa_kl0.01
#   TRAIN_REWARD=scienceqa_kl0.01 STEP=700 bash run_eval_understanding_scienceqa.sh
#
#   # Explicit LoRA path
#   LORA_PATH=/abs/path/to/lora_adapter_ema bash run_eval_understanding_scienceqa.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
source /your_path/miniconda3/etc/profile.d/conda.sh
conda activate dflow_grpo

export PYTHONNOUSERSITE=1
export PYTHONPATH=/your_path/dFlowGRPO/FUDOKI:/your_path/dFlowGRPO/flow_grpo:/your_path/dFlowGRPO/discrete_flow_grpo:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# =============================================================================
# >>>>>>>>>> CUSTOMIZE HERE <<<<<<<<<<
# =============================================================================

# Put your OpenAI key here (or export OPENAI_API_KEY before calling the script).
_DEFAULT_OPENAI_API_KEY="your_key_here"

CKPT_PATH=${CKPT_PATH:-/your_path/dFlowGRPO/FUDOKI/checkpoints}
SPLIT=${SPLIT:-test}
STEPS=${STEPS:-20}
NUM_SAMPLES=${NUM_SAMPLES:-0}
API_MODEL=${API_MODEL:-gpt-4o-mini}
NUM_GPUS=${NUM_GPUS:-8}
ACC_CONFIG=${ACC_CONFIG:-${SCRIPT_DIR}/accelerate_config_ds.yaml}

# ---- Checkpoint selection (run_eval.sh style) ----
TRAIN_REWARD=${TRAIN_REWARD:-scienceqa_kl0.01}
GRPO_DIR=${GRPO_DIR:-${SCRIPT_DIR}/output_grpo_u_${TRAIN_REWARD}}
STEP=${STEP:-}                  # e.g. 700 ; empty -> baseline (unless LORA_PATH set)
USE_EMA=${USE_EMA:-1}           # 1 -> lora_adapter_ema, 0 -> lora_adapter

# ---- BATCH mode (mirrors run_eval.sh) ----
BATCH=${BATCH:-1}                            # 0 = single run, 1 = sweep checkpoints
STEP_INTERVAL=${STEP_INTERVAL:-"100:200"}    # "start:stride[:end]" or plain int
EMA_ONLY=${EMA_ONLY:-1}                      # 1 = eval only lora_adapter_ema variants
SKIP_EXISTING=${SKIP_EXISTING:-1}            # 1 = skip if output dir already exists
SKIP_BASELINE=${SKIP_BASELINE:-0}            # 1 = skip no-LoRA baseline run in batch

# Explicit overrides (win over STEP-based auto-discovery).
LORA_PATH=${LORA_PATH:-}
EMA_PATH=${EMA_PATH:-}
GEN_MODULES_PATH=${GEN_MODULES_PATH:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-${_DEFAULT_OPENAI_API_KEY}}
API_BASE_URL=${API_BASE_URL:-}
OUTPUT_DIR=${OUTPUT_DIR:-}

# =============================================================================

# ---- Logging: tee stdout+stderr to logs/ -----------------------------------
# Override LOG_FILE=... explicitly, or NO_LOG=1 to disable.
NO_LOG=${NO_LOG:-0}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
if [ "${NO_LOG}" != "1" ]; then
    mkdir -p "${LOG_DIR}"
    _ts=$(date +%Y%m%d_%H%M%S)
    _mode_tag=$([ "${BATCH:-0}" = "1" ] && echo "batch" || echo "single")
    LOG_FILE=${LOG_FILE:-${LOG_DIR}/eval_scienceqa_${TRAIN_REWARD}_${_mode_tag}_${_ts}.log}
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "[run_eval_understanding_scienceqa.sh] logging to ${LOG_FILE}"
fi

# ---- Parse STEP_INTERVAL = "start:stride[:end]" or plain int (run_eval.sh) ----
_SI_STRIDE=""; _SI_START=""; _SI_END=""
if [ -n "${STEP_INTERVAL}" ]; then
    if [[ "${STEP_INTERVAL}" == *:* ]]; then
        _SI_START=$(echo "${STEP_INTERVAL}" | cut -d: -f1)
        _SI_STRIDE=$(echo "${STEP_INTERVAL}" | cut -d: -f2)
        _SI_END=$(echo   "${STEP_INTERVAL}" | cut -d: -f3)
        [ -z "${_SI_START}" ] && _SI_START=0
        if [ -z "${_SI_STRIDE}" ] || ! [ "${_SI_STRIDE}" -gt 0 ] 2>/dev/null; then
            echo "ERROR: STEP_INTERVAL='${STEP_INTERVAL}' has invalid stride '${_SI_STRIDE}'"
            exit 1
        fi
    else
        _SI_STRIDE="${STEP_INTERVAL}"
        _SI_START=0
    fi
fi

# ---- Helper: run eval_scienceqa.py once with the current ARGS array ----
_launch_one() {
    if [ "${NUM_GPUS}" -le 1 ]; then
        python "${SCRIPT_DIR}/eval_scienceqa.py" "${ARGS[@]}" "$@"
    else
        accelerate launch \
            --num_processes "${NUM_GPUS}" \
            --num_machines 1 \
            --mixed_precision no \
            --dynamo_backend no \
            "${SCRIPT_DIR}/eval_scienceqa.py" "${ARGS[@]}" "$@"
    fi
}

# ---- Helper: build ARGS for a given (LORA_PATH, OUTPUT_DIR) and run ----
_build_and_run() {
    local _lora="$1"
    local _out="$2"
    local _ema="${3:-}"

    ARGS=(
        --checkpoint_path "${CKPT_PATH}"
        --split "${SPLIT}"
        --steps "${STEPS}"
        --num_samples "${NUM_SAMPLES}"
        --api_model "${API_MODEL}"
    )
    [ -n "${_lora}" ]            && ARGS+=(--lora_path "${_lora}")
    [ -n "${_ema}" ]             && ARGS+=(--ema_path "${_ema}")
    [ -n "${GEN_MODULES_PATH}" ] && ARGS+=(--gen_modules_path "${GEN_MODULES_PATH}")
    if [ -n "${OPENAI_API_KEY}" ] && [ "${OPENAI_API_KEY}" != "sk-REPLACE_ME" ]; then
        ARGS+=(--api_key "${OPENAI_API_KEY}")
        export OPENAI_API_KEY
    fi
    [ -n "${API_BASE_URL}" ]     && ARGS+=(--api_base_url "${API_BASE_URL}")
    [ -n "${_out}" ]             && ARGS+=(--output_dir "${_out}")

    echo ""
    echo "===== [scienceqa] LORA=${_lora:-<baseline>}  OUT=${_out} ====="
    # NOTE: do NOT forward "$@" here -- our own positional args ($1=lora,
    # $2=out, $3=ema) would get appended as extra CLI flags to
    # eval_scienceqa.py and trip "unrecognized arguments".
    _launch_one
}

# =============================================================================
# BATCH MODE: sweep checkpoints in GRPO_DIR matching STEP_INTERVAL
# =============================================================================
if [ "${BATCH}" -eq 1 ] 2>/dev/null; then
    echo "[run_eval_understanding_scienceqa.sh] BATCH mode"
    echo "  GRPO_DIR=${GRPO_DIR}"
    echo "  STEP_INTERVAL=${STEP_INTERVAL}  (start=${_SI_START} stride=${_SI_STRIDE} end=${_SI_END:-max})"
    echo "  EMA_ONLY=${EMA_ONLY}  SKIP_EXISTING=${SKIP_EXISTING}  SKIP_BASELINE=${SKIP_BASELINE}"

    if [ ! -d "${GRPO_DIR}" ]; then
        echo "ERROR: GRPO_DIR not found: ${GRPO_DIR}"
        exit 1
    fi

    _BASE_OUT="${SCRIPT_DIR}/eval_scienceqa_${TRAIN_REWARD}_results"
    mkdir -p "${_BASE_OUT}"

    # ---- Helper: did a previous run for this output dir actually finish?
    # eval_scienceqa.py writes summary.json only at the very end, so its
    # presence is the only reliable "done" marker. An empty dir means the
    # previous attempt crashed before producing any results -- we should
    # NOT skip it.
    _is_complete() {
        [ -f "$1/summary.json" ]
    }

    # Optional baseline (no LoRA) once.
    if [ "${SKIP_BASELINE}" -ne 1 ] 2>/dev/null; then
        _baseline_out="${_BASE_OUT}/baseline_${STEPS}nfe_${SPLIT}"
        if [ "${SKIP_EXISTING}" -eq 1 ] 2>/dev/null && _is_complete "${_baseline_out}"; then
            echo "  [skip] baseline already complete: ${_baseline_out}"
        else
            if [ -d "${_baseline_out}" ] && ! _is_complete "${_baseline_out}"; then
                echo "  [redo] baseline dir exists but no summary.json -- re-running"
            fi
            _build_and_run "" "${_baseline_out}" ""
        fi
    fi

    # Discover all checkpoint_*_step<N> dirs and iterate in numeric order.
    mapfile -t _ckpts < <(
        find "${GRPO_DIR}" -maxdepth 1 -type d -name "checkpoint_*_step*" 2>/dev/null \
        | awk -F_step '{print $NF" "$0}' \
        | sort -n -k1,1 \
        | awk '{print $2}'
    )

    if [ "${#_ckpts[@]}" -eq 0 ]; then
        echo "WARNING: no checkpoint_*_step* dirs under ${GRPO_DIR}"
    fi

    for _ckpt_dir in "${_ckpts[@]}"; do
        _step="${_ckpt_dir##*_step}"
        # Numeric sanity check
        if ! [[ "${_step}" =~ ^[0-9]+$ ]]; then
            continue
        fi
        # Filter by start / stride / end
        if [ "${_step}" -lt "${_SI_START:-0}" ]; then continue; fi
        if [ -n "${_SI_END}" ] && [ "${_step}" -gt "${_SI_END}" ]; then continue; fi
        if [ -n "${_SI_STRIDE}" ] && [ "${_SI_STRIDE}" -gt 0 ]; then
            _delta=$(( _step - ${_SI_START:-0} ))
            if [ $(( _delta % _SI_STRIDE )) -ne 0 ]; then
                continue
            fi
        fi

        # NOTE: eval_scienceqa.py uses a flat layout:
        #   <ckpt>/lora_adapter/        <- the LoRA weights
        #   <ckpt>/ema_state.pt         <- EMA weights (auto-detected next
        #                                  to lora_adapter)
        # There is NO separate `lora_adapter_ema/` dir. EMA is applied
        # automatically by eval_scienceqa.py when ema_state.pt exists, so
        # we just point at lora_adapter/ and tag the output dir.
        if [ ! -d "${_ckpt_dir}/lora_adapter" ]; then
            echo "  [skip] step${_step}: no lora_adapter/ in ${_ckpt_dir}"
            continue
        fi

        _has_ema=0
        [ -f "${_ckpt_dir}/ema_state.pt" ] && _has_ema=1

        if [ "${EMA_ONLY}" -eq 1 ] 2>/dev/null && [ "${_has_ema}" -ne 1 ]; then
            echo "  [skip] step${_step}: EMA_ONLY=1 but no ema_state.pt"
            continue
        fi

        _suffix="step${_step}"
        [ "${_has_ema}" -eq 1 ] && _suffix="${_suffix}_ema"
        _out="${_BASE_OUT}/${_suffix}"
        if [ "${SKIP_EXISTING}" -eq 1 ] 2>/dev/null && _is_complete "${_out}"; then
            echo "  [skip] ${_suffix} already complete"
        else
            if [ -d "${_out}" ] && ! _is_complete "${_out}"; then
                echo "  [redo] ${_suffix} exists but no summary.json -- re-running"
            fi
            _build_and_run "${_ckpt_dir}/lora_adapter" "${_out}" ""
        fi
    done

    echo ""
    echo "[run_eval_understanding_scienceqa.sh] BATCH eval complete. Results: ${_BASE_OUT}"
    exit 0
fi

# =============================================================================
# SINGLE MODE (original behavior)
# =============================================================================

# ---- Auto-discover LoRA adapter from STEP if LORA_PATH is not set ----
if [ -z "${LORA_PATH}" ] && [ -n "${STEP}" ]; then
    _ckpt_dir=$(find "${GRPO_DIR}" -maxdepth 1 -type d -name "checkpoint_*_step${STEP}" 2>/dev/null | head -1)
    if [ -z "${_ckpt_dir}" ]; then
        echo "ERROR: No checkpoint found for step ${STEP} in ${GRPO_DIR}"
        echo "Available checkpoints:"
        ls -d "${GRPO_DIR}"/checkpoint_*_step* 2>/dev/null | head -20 || echo "  (none)"
        exit 1
    fi
    if [ "${USE_EMA}" -eq 1 ] 2>/dev/null && [ -d "${_ckpt_dir}/lora_adapter_ema" ]; then
        LORA_PATH="${_ckpt_dir}/lora_adapter_ema"
    elif [ -d "${_ckpt_dir}/lora_adapter" ]; then
        LORA_PATH="${_ckpt_dir}/lora_adapter"
    else
        echo "ERROR: No lora_adapter[_ema] dir under ${_ckpt_dir}"
        exit 1
    fi
    # Auto output dir: eval_scienceqa_<TRAIN_REWARD>_results/step<STEP>[_ema]
    # NOTE: scienceqa training stores EMA as `ema_state.pt` next to
    # `lora_adapter/` (no separate lora_adapter_ema dir). eval_scienceqa.py
    # auto-loads ema_state.pt when present, so we tag the output dir _ema
    # whenever that file exists.
    if [ -z "${OUTPUT_DIR}" ]; then
        _suffix="step${STEP}"
        if [ "${USE_EMA}" -eq 1 ] 2>/dev/null \
                && { [ -d "${_ckpt_dir}/lora_adapter_ema" ] \
                     || [ -f "${_ckpt_dir}/ema_state.pt" ]; }; then
            _suffix="${_suffix}_ema"
        fi
        OUTPUT_DIR="${SCRIPT_DIR}/eval_scienceqa_${TRAIN_REWARD}_results/${_suffix}"
    fi
fi

# Baseline (no LoRA, no STEP): still put it under the same results tree so
# baseline and LoRA runs live side-by-side.
if [ -z "${LORA_PATH}" ] && [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${SCRIPT_DIR}/eval_scienceqa_${TRAIN_REWARD}_results/baseline_${STEPS}nfe_${SPLIT}"
fi

ARGS=(
    --checkpoint_path "${CKPT_PATH}"
    --split "${SPLIT}"
    --steps "${STEPS}"
    --num_samples "${NUM_SAMPLES}"
    --api_model "${API_MODEL}"
)
[ -n "${LORA_PATH}" ]         && ARGS+=(--lora_path "${LORA_PATH}")
[ -n "${EMA_PATH}" ]          && ARGS+=(--ema_path "${EMA_PATH}")
[ -n "${GEN_MODULES_PATH}" ]  && ARGS+=(--gen_modules_path "${GEN_MODULES_PATH}")
if [ -n "${OPENAI_API_KEY}" ] && [ "${OPENAI_API_KEY}" != "sk-REPLACE_ME" ]; then
    ARGS+=(--api_key "${OPENAI_API_KEY}")
    export OPENAI_API_KEY
    _api_key_state="set"
else
    _api_key_state="UNSET (placeholder) -- judge will be skipped / will error"
fi
[ -n "${API_BASE_URL}" ]      && ARGS+=(--api_base_url "${API_BASE_URL}")
[ -n "${OUTPUT_DIR}" ]        && ARGS+=(--output_dir "${OUTPUT_DIR}")

echo "[run_eval_understanding_scienceqa.sh] NUM_GPUS=${NUM_GPUS}  CKPT=${CKPT_PATH}"
echo "[run_eval_understanding_scienceqa.sh] TRAIN_REWARD=${TRAIN_REWARD}  STEP=${STEP:-<none>}  USE_EMA=${USE_EMA}"
echo "[run_eval_understanding_scienceqa.sh] GRPO_DIR=${GRPO_DIR}"
echo "[run_eval_understanding_scienceqa.sh] SPLIT=${SPLIT} STEPS=${STEPS} NUM_SAMPLES=${NUM_SAMPLES}"
echo "[run_eval_understanding_scienceqa.sh] LORA=${LORA_PATH:-<baseline>}  EMA=${EMA_PATH:-<auto>}"
echo "[run_eval_understanding_scienceqa.sh] OUTPUT_DIR=${OUTPUT_DIR:-<auto>}"
echo "[run_eval_understanding_scienceqa.sh] API=${API_MODEL}  api_key=${_api_key_state}"

if [ "${NUM_GPUS}" -le 1 ]; then
    python "${SCRIPT_DIR}/eval_scienceqa.py" "${ARGS[@]}" "$@"
else
    # Plain multi-GPU DDP -- no DeepSpeed needed for eval.
    accelerate launch \
        --num_processes "${NUM_GPUS}" \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        "${SCRIPT_DIR}/eval_scienceqa.py" "${ARGS[@]}" "$@"
fi

