#!/usr/bin/env bash
# filepath: /your_path/dFlowGRPO/discrete_flow_grpo/run_eval_understanding.sh
#
# Run VLMEvalKit evaluation of the FUDOKI understanding model on multiple
# benchmarks: POPE, MME(-P), MMBench_DEV_EN, SEEDBench_IMG,
# GQA_TestDev_Balanced, MMMU_DEV_VAL, MMVet.
#
# Usage examples:
#   # Base FUDOKI only
#   bash run_eval_understanding.sh
#
#   # With RL-trained LoRA (+ auto-detected EMA next to it) — direct paths
#   LORA_PATH=/path/to/output_grpo_u_scienceqa/checkpoint_step_XXX/lora_adapter \
#   bash run_eval_understanding.sh
#
#   # Or, run_eval.sh-style: scan output_grpo_<TRAIN_REWARD>/ for step <STEP>
#   TRAIN_REWARD=u_scienceqa_kl0.01 STEP=1700 \
#   bash run_eval_understanding.sh
#
#   # Pick a subset of benchmarks
#   BENCHMARKS="POPE MME" bash run_eval_understanding.sh
#
#   # Override number of GPUs / steps
#   NPROC=4 STEPS=64 bash run_eval_understanding.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Conda env (mirror run_eval.sh — system python lacks VLMEvalKit deps).
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-fudoki_vlm}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    else
        echo "[run_eval_understanding.sh] WARNING: conda.sh not found; assuming ${CONDA_ENV} is already active."
    fi
fi
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# PYTHONPATH (mirror run_eval.sh — VLMEvalKit's run.py imports `fudoki.*`).
# ---------------------------------------------------------------------------
_FUDOKI_ROOT="${FUDOKI_ROOT:-/your_path/dFlowGRPO/FUDOKI}"
_FLOW_GRPO_ROOT="${FLOW_GRPO_ROOT:-/your_path/dFlowGRPO/flow_grpo}"
_GRPO_PKG_ROOT="${GRPO_PKG_ROOT:-/your_path/dFlowGRPO/discrete_flow_grpo}"
export PYTHONPATH="${_FUDOKI_ROOT}:${_FLOW_GRPO_ROOT}:${_GRPO_PKG_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Paths (override via environment if needed)
# ---------------------------------------------------------------------------
CKPT_PATH="${CKPT_PATH:-/your_path/dFlowGRPO/FUDOKI/checkpoints}"
GRPO_ROOT="${GRPO_ROOT:-/your_path/dFlowGRPO/discrete_flow_grpo}"
VLM_ROOT="${VLM_ROOT:-${GRPO_ROOT}/dataset_understanding/VLMEvalKit}"

OUTPUT_DIR="${OUTPUT_DIR:-${VLM_ROOT}/fudoki_output/understanding}"
MODEL_TAG="${MODEL_TAG:-understanding}"

# Optional RL-trained weights
LORA_PATH="${LORA_PATH:-}"
EMA_PATH="${EMA_PATH:-}"
GEN_MODULES_PATH="${GEN_MODULES_PATH:-}"

# ---------------------------------------------------------------------------
# run_eval.sh-style resolution: TRAIN_REWARD + STEP -> LORA_PATH / EMA_PATH.
# Looks under ${GRPO_ROOT}/output_grpo_<TRAIN_REWARD>/checkpoint_*_step<STEP>/.
# Picks lora_adapter_ema if present (USE_EMA=1, default), else lora_adapter.
# ---------------------------------------------------------------------------
TRAIN_REWARD="${TRAIN_REWARD:-u_scienceqa_kl0.01}"
STEP="${STEP:-1500}"
USE_EMA="${USE_EMA:-1}"
GRPO_DIR="${GRPO_DIR:-}"
if [[ -z "${LORA_PATH}" && -n "${TRAIN_REWARD}" && -n "${STEP}" ]]; then
    [[ -z "${GRPO_DIR}" ]] && GRPO_DIR="${GRPO_ROOT}/output_grpo_${TRAIN_REWARD}"
    if [[ ! -d "${GRPO_DIR}" ]]; then
        echo "[run_eval_understanding.sh] ERROR: GRPO_DIR not found: ${GRPO_DIR}" >&2
        exit 1
    fi
    _ckpt_dir="$(find "${GRPO_DIR}" -maxdepth 1 -type d -name "checkpoint_*_step${STEP}" 2>/dev/null | head -1)"
    if [[ -z "${_ckpt_dir}" ]]; then
        echo "[run_eval_understanding.sh] ERROR: No checkpoint_*_step${STEP} under ${GRPO_DIR}" >&2
        echo "Available:" >&2
        ls -d "${GRPO_DIR}"/checkpoint_*_step* 2>/dev/null | head -20 >&2 || true
        exit 1
    fi
    if [[ "${USE_EMA}" == "1" && -d "${_ckpt_dir}/lora_adapter_ema" ]]; then
        LORA_PATH="${_ckpt_dir}/lora_adapter_ema"
    elif [[ -d "${_ckpt_dir}/lora_adapter" ]]; then
        LORA_PATH="${_ckpt_dir}/lora_adapter"
    else
        echo "[run_eval_understanding.sh] ERROR: no lora_adapter[_ema] in ${_ckpt_dir}" >&2
        exit 1
    fi
    echo "[run_eval_understanding.sh] resolved LORA_PATH=${LORA_PATH}"
fi

# ---------------------------------------------------------------------------
# OpenAI API key for the judge models (MMMU / MMVet / chatgpt-0125 / ...).
# Priority: existing $OPENAI_API_KEY in the env > _DEFAULT_OPENAI_API_KEY here.
# Replace the placeholder below with your own key, or just
#   export OPENAI_API_KEY=sk-...
# before invoking this script.
# ---------------------------------------------------------------------------
_DEFAULT_OPENAI_API_KEY="your_openai_api_key_here"
OPENAI_API_KEY="${OPENAI_API_KEY:-${_DEFAULT_OPENAI_API_KEY}}"
# Optional: override base URL (e.g. for an OpenAI-compatible proxy).
OPENAI_API_BASE="${OPENAI_API_BASE:-}"

if [[ -z "${OPENAI_API_KEY}" || "${OPENAI_API_KEY}" == "sk-REPLACE_ME" ]]; then
    echo "[run_eval_understanding.sh] WARNING: OPENAI_API_KEY is not set."
    echo "    Judge-based benchmarks (MMMU / MMVet / chatgpt-0125) will fail."
else
    export OPENAI_API_KEY
    [[ -n "${OPENAI_API_BASE}" ]] && export OPENAI_API_BASE
fi

# Auto-detect EMA state dict next to the LoRA adapter if not provided.
if [[ -n "${LORA_PATH}" && -z "${EMA_PATH}" ]]; then
    _cand="$(dirname "${LORA_PATH}")/ema_state.pt"
    if [[ -f "${_cand}" ]]; then
        EMA_PATH="${_cand}"
        echo "[run_eval_understanding.sh] auto-detected EMA: ${EMA_PATH}"
    fi
fi

# Append LoRA/EMA tag to the output dir so runs don't collide.
if [[ -n "${LORA_PATH}" ]]; then
    _lora_stamp="$(basename "$(dirname "${LORA_PATH}")")_$(basename "${LORA_PATH}")"
    MODEL_TAG="${MODEL_TAG}_${_lora_stamp}"
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Logging: tee stdout+stderr to logs/<MODEL_TAG>_<timestamp>.log.
# Override with LOG_FILE=... or disable with NO_LOG=1.
# ---------------------------------------------------------------------------
NO_LOG="${NO_LOG:-0}"
# Logs go under the output dir so they live alongside the eval results.
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
if [[ "${NO_LOG}" != "1" ]]; then
    mkdir -p "${LOG_DIR}"
    _ts="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="${LOG_FILE:-${LOG_DIR}/eval_understanding_${MODEL_TAG}_${_ts}.log}"
    # Redirect everything below this line to both terminal and log file.
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "[run_eval_understanding.sh] logging to ${LOG_FILE}"
fi

# ---------------------------------------------------------------------------
# Runtime knobs
# ---------------------------------------------------------------------------
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-12358}"
STEPS="${STEPS:-32}"          # discrete-flow-matching denoising steps
TXT_MAX_LENGTH="${TXT_MAX_LENGTH:-500}"
SEED="${SEED:-99}"
JUDGE="${JUDGE:-chatgpt-0125}"
# MMMU: FUDOKI's run_local.sh uses gpt-4-0125 (→ 'gpt-4-0125-preview' in
# VLMEvalKit). That snapshot is retired by OpenAI; the closest still-served
# GPT-4-class judge is gpt-4-turbo (= gpt-4-turbo-2024-04-09 after our
# judge_util.py patch).
JUDGE_MMMU="${JUDGE_MMMU:-gpt-4-0125}"
# MMVet: FUDOKI's run_local.sh does NOT pass --judge -> defaults to gpt-4-turbo
# inside VLMEvalKit (which originally pointed at 'gpt-4-1106-preview', also
# retired). We use gpt-4-turbo, redirected to gpt-4-turbo-2024-04-09.
JUDGE_MMVET="${JUDGE_MMVET:-gpt-4-turbo}"

# Parallel OpenAI API calls during the judge step. Lower this if you hit
# 429 rate-limit errors from OpenAI (e.g. tier-1 keys have 30k TPM on
# gpt-4-turbo, which only supports ~1 worker). Default 1 to be safe.
API_NPROC="${API_NPROC:-1}"
# How many retries the OpenAI judge wrapper will do on 429 / parse failures
# before giving up on a sample. VLMEvalKit default is 3; bump to 8 to absorb
# transient rate-limits and the occasional malformed gpt-4-turbo response.
JUDGE_RETRY="${JUDGE_RETRY:-8}"
# Min seconds between successive judge API calls (process-wide rate limit
# applied by our patched judge_util.py). Empty -> auto: ~7.5s for gpt-4*,
# ~0.6s for gpt-3.5/chatgpt-0125. Override e.g. JUDGE_MIN_INTERVAL=10.
JUDGE_MIN_INTERVAL="${JUDGE_MIN_INTERVAL:-}"
if [[ -n "${JUDGE_MIN_INTERVAL}" ]]; then
    export JUDGE_MIN_INTERVAL
fi
# GQA's evaluator forks an mp.Pool of judge workers (default 16); each fork
# has its own OpenAI client and bypasses our process-wide rate limiter, so
# 16 forks easily blow past chatgpt-0125's 500 RPM tier-1 cap and most calls
# come back as 429 -> fail_msg -> 0 score. The patched image_vqa.py reads
# JUDGE_POOL_SIZE; defaults to 1 for GQA (rate-limit safe, ~25 min for 12.5k
# samples) and 16 for non-API string matchers.
JUDGE_POOL_SIZE="${JUDGE_POOL_SIZE:-4}"
if [[ -n "${JUDGE_POOL_SIZE}" ]]; then
    export JUDGE_POOL_SIZE
fi

# Benchmarks to run (space-separated). Default is the full requested set.
BENCHMARKS_DEFAULT="MMMU_DEV_VAL MMVet"
# POPE MME MMBench_DEV_EN SEEDBench_IMG GQA_TestDev_Balanced MMMU_DEV_VAL MMVet
BENCHMARKS="${BENCHMARKS:-${BENCHMARKS_DEFAULT}}"

# Per-benchmark step-count overrides (mirrors FUDOKI's run_local.sh).
declare -A STEPS_MAP=(
    ["POPE"]="${STEPS}"
    ["MME"]="${STEPS}"
    ["MMBench_DEV_EN"]="${STEPS}"
    ["SEEDBench_IMG"]="${STEPS}"
    ["GQA_TestDev_Balanced"]="${GQA_STEPS:-20}"
    ["MMMU_DEV_VAL"]="${STEPS}"
    ["MMVet"]="${MMVET_STEPS:-100}"
)

# Per-benchmark judge overrides.
declare -A JUDGE_MAP=(
    ["POPE"]="${JUDGE}"
    ["MME"]="${JUDGE}"
    ["MMBench_DEV_EN"]="${JUDGE}"
    ["SEEDBench_IMG"]="${JUDGE}"
    ["GQA_TestDev_Balanced"]="${JUDGE}"
    ["MMMU_DEV_VAL"]="${JUDGE_MMMU}"
    ["MMVet"]="${JUDGE_MMVET}"
)

# ---------------------------------------------------------------------------
# Assemble extra args for LoRA / EMA / gen_modules.
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
if [[ -n "${LORA_PATH}" ]]; then
    EXTRA_ARGS+=(--lora_path "${LORA_PATH}")
fi
if [[ -n "${EMA_PATH}" ]]; then
    EXTRA_ARGS+=(--ema_path "${EMA_PATH}")
fi
if [[ -n "${GEN_MODULES_PATH}" ]]; then
    EXTRA_ARGS+=(--gen_modules_path "${GEN_MODULES_PATH}")
fi

echo "==========================================================="
echo " FUDOKI understanding eval"
echo "   CKPT:       ${CKPT_PATH}"
echo "   MODEL_TAG:  ${MODEL_TAG}"
echo "   OUTPUT_DIR: ${OUTPUT_DIR}"
echo "   NPROC:      ${NPROC}"
echo "   STEPS:      ${STEPS}"
echo "   LoRA:       ${LORA_PATH:-<none>}"
echo "   EMA:        ${EMA_PATH:-<none>}"
echo "   GEN:        ${GEN_MODULES_PATH:-<none>}"
echo "   BENCHMARKS: ${BENCHMARKS}"
echo "   OPENAI_API: $([[ -n "${OPENAI_API_KEY:-}" && "${OPENAI_API_KEY}" != "sk-REPLACE_ME" ]] && echo 'set' || echo 'UNSET')${OPENAI_API_BASE:+  base=${OPENAI_API_BASE}}"
echo "==========================================================="

# Reuse previous predictions if available (skips re-inference, runs only the
# judge step). Set REUSE=0 to force a fresh inference run.
REUSE="${REUSE:-1}"
REUSE_ARGS=()
if [[ "${REUSE}" == "1" ]]; then
    REUSE_ARGS+=(--reuse)
fi

# Per-benchmark judge call statistics (final fails / 429s / other HTTP errs)
# get appended here as JSON lines by our patched judge_util.py.
JUDGE_STATS_FILE="${JUDGE_STATS_FILE:-${OUTPUT_DIR}/${MODEL_TAG}/judge_stats.jsonl}"
mkdir -p "$(dirname "${JUDGE_STATS_FILE}")"
export JUDGE_STATS_FILE

cd "${VLM_ROOT}"

for DATA in ${BENCHMARKS}; do
    _steps="${STEPS_MAP[${DATA}]:-${STEPS}}"
    _judge="${JUDGE_MAP[${DATA}]:-${JUDGE}}"

    echo ""
    echo "----- [${DATA}] steps=${_steps} judge=${_judge:-<default>} -----"

    # Only pass --judge if non-empty (FUDOKI's run_local.sh omits --judge for MMVet).
    JUDGE_ARGS=()
    if [[ -n "${_judge}" ]]; then
        JUDGE_ARGS+=(--judge "${_judge}")
    fi

    # Tag judge stats so each benchmark gets its own row in JUDGE_STATS_FILE.
    export JUDGE_BENCHMARK="${DATA}"

    torchrun \
        --nproc_per_node "${NPROC}" --master-port "${MASTER_PORT}" \
        run.py --data "${DATA}" --model "${MODEL_TAG}" \
        --checkpoint_path "${CKPT_PATH}" \
        --text_embedding_path "${CKPT_PATH}/text_embedding.pt" \
        --image_embedding_path "${CKPT_PATH}/image_embedding.pt" \
        --discrete_fm_steps "${_steps}" \
        --output_dir "${OUTPUT_DIR}" \
        "${JUDGE_ARGS[@]}" \
        "${REUSE_ARGS[@]}" \
        --judge-args '{"verbose": true}' \
        --api-nproc "${API_NPROC}" \
        --retry "${JUDGE_RETRY}" \
        --txt_max_length "${TXT_MAX_LENGTH}" \
        --seed "${SEED}" \
        "${EXTRA_ARGS[@]}"
done

echo ""
echo "All benchmarks finished. Results under: ${OUTPUT_DIR}/${MODEL_TAG}"
