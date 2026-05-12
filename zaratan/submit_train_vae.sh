#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-128G}"

export REPO_ROOT NUM_GPUS
export VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B}"
export OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/vae_model}"
export RUN_NAME="${RUN_NAME:-gsm8k-vae}"
export EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${LOG_DIR}"

sbatch_args=(
  "--job-name=${RUN_NAME}"
  "--nodes=1" "--ntasks=1"
  "--chdir=${REPO_ROOT}"
  "--cpus-per-task=${CPUS_PER_TASK}"
  "--time=${TIME_LIMIT}"
  "--mem=${MEMORY}"
  "--output=${LOG_DIR}/%x-%j.out"
  "--export=ALL"
)
if [[ -n "${ACCOUNT}" ]]; then sbatch_args+=("--account=${ACCOUNT}"); fi
if [[ -n "${PARTITION}" ]]; then sbatch_args+=("--partition=${PARTITION}"); fi
if [[ -n "${QOS}" ]]; then sbatch_args+=("--qos=${QOS}"); fi
if [[ -n "${GPU_TYPE}" ]]; then sbatch_args+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}"); else sbatch_args+=("--gres=gpu:${NUM_GPUS}"); fi

sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_train_vae.sbatch"
