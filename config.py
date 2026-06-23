"""
Configuration classes for the VAE-based diffusion model.
"""

from dataclasses import dataclass
from typing import Optional

DEFAULT_LLM_MODEL = "/scratch/zt1/project/kaiqing-prj/user/mbezick/ladir/models/Llama-3.1-8B"

@dataclass
class LMFusionConfig:
    """Configuration class for LMFusion model."""
    hidden_size: int = 768
    model_name_or_path: str = DEFAULT_LLM_MODEL
    use_flash_attn: bool = True
    freeze_tokenizer: bool = True
    discrete: bool = False
    use_query_embeddings: bool = False
    num_codebooks: int = 8
    patch_embedding_length: int = 64
    block: bool = False
    cos_loss: float = 1.0
    emu2_expr: bool = False
    subsampling: bool = False
