# Zaratan GSM8K Jobs

Run `python prepare_gsm8k_data.py` before submitting training jobs.

Recommended order:

1. Train the VAE:
   ```bash
   PARTITION=gpu NUM_GPUS=8 zaratan/submit_train_vae.sh
   ```

2. Update `configs/cd_formal_8B_VAE_conn.yaml` so `ae.icae_ckpt` points at the
   trained VAE checkpoint, then train diffusion:
   ```bash
   PARTITION=gpu NUM_GPUS=8 zaratan/submit_train_diffusion.sh
   ```

3. Generate scorer data from a trained diffusion checkpoint:
   ```bash
   DIFFUSION_CKPT=ckpt/vae_diffusion_experiment/checkpoint-2500 \
     PARTITION=gpu zaratan/submit_generate_scorer_data.sh
   ```

4. Train the latent scorer:
   ```bash
   PARTITION=gpu zaratan/submit_train_scorer.sh
   ```

Common overrides:

- `VENV_PATH=/path/to/venv`
- `HF_HOME=/path/to/hf_cache`
- `LLM_MODEL=/path/to/local/Llama-3.1-8B`
- `ACCOUNT=... PARTITION=... QOS=... GPU_TYPE=a100`
- `EXTRA_ARGS="trainer.learning_rate=5e-5"` for diffusion config overrides.
