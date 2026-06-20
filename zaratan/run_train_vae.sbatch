#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/vae_model}"
RUN_NAME="${RUN_NAME:-dart-math-vae}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
MASTER_PORT="${MASTER_PORT:-46079}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"
source "${VENV_PATH}/bin/activate"
python zaratan/check_python_env.py

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cmd=(
  python -m torch.distributed.run
  --nproc_per_node "${NUM_GPUS}"
  --master_port "${MASTER_PORT}"
  vae/train_vae.py
  --run_name "${RUN_NAME}"
  --model_name_or_path "${LLM_MODEL}"
  --lora_r 512
  --lora_alpha 256
  --lora_dropout 0.05
  --output_dir "${OUTPUT_DIR}"
  --input_type full_format
  --test_size 500
  --max_steps 30000
  --num_train_epochs 10
  --learning_rate 2e-5
  --lr_scheduler_type cosine
  --lr_scheduler_kwargs '{"num_cycles": 1}'
  --warmup_steps 1000
  --optim adamw_torch
  --weight_decay 0.03
  --eval_strategy no
  --save_strategy steps
  --save_steps 2500
  --ddp_backend nccl
  --fsdp 'hybrid_shard auto_wrap'
  --fsdp_config '{"backward_prefetch": "backward_pre", "forward_prefetch": true, "cpu_ram_efficient_loading": true, "sync_module_states": true, "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"], "use_orig_params": true, "activation_checkpointing": false}'
  --notes "DART-Math VAE training"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  cmd+=("${extra_args[@]}")
fi

printf 'Running: %s\n' "${cmd[*]}"
srun "${cmd[@]}"
