from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentScorerConfig:
    vocab_size: int
    latent_dim: int = 128
    hidden_size: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_timestep: int = 1000
    pad_token_id: int = 0
    reward_eps: float = 1e-6


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.float()
        half = self.hidden_size // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(float(self.max_period), device=timestep.device))
            * torch.arange(half, device=timestep.device).float()
            / max(half, 1)
        )
        args = timestep[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.hidden_size:
            emb = F.pad(emb, (0, self.hidden_size - emb.shape[-1]))
        return self.proj(emb)


class LatentReasoningScorer(nn.Module):
    """Scores whether a question-conditioned latent reasoning state will decode correctly."""

    def __init__(self, config: LatentScorerConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.time_embed = SinusoidalTimestepEmbedding(config.hidden_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.type_embed = nn.Embedding(3, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, 1)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, input_ids_q: torch.Tensor, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        batch_size, num_latents, _ = x_t.shape
        timestep = torch.as_tensor(timestep, device=x_t.device).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)

        q_emb = self.token_embed(input_ids_q.to(x_t.device))
        q_emb = q_emb + self.type_embed.weight[0].view(1, 1, -1)
        t_emb = self.time_embed(timestep).unsqueeze(1) + self.type_embed.weight[1].view(1, 1, -1)
        z_emb = self.latent_proj(x_t.float()) + self.type_embed.weight[2].view(1, 1, -1)
        cls = self.cls.expand(batch_size, -1, -1)
        hidden = torch.cat([cls, q_emb, t_emb, z_emb], dim=1)

        q_pad = input_ids_q.to(x_t.device).eq(self.config.pad_token_id)
        nonpad = torch.zeros(batch_size, 1 + 1 + num_latents, dtype=torch.bool, device=x_t.device)
        key_padding_mask = torch.cat(
            [nonpad[:, :1], q_pad, nonpad[:, 1:]],
            dim=1,
        )

        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        logits = self.head(self.norm(encoded[:, 0])).squeeze(-1)
        return logits

    def get_reward(self, input_ids_q: torch.Tensor, x_t: torch.Tensor, timestep: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
        if timestep is None:
            timestep = torch.zeros(x_t.shape[0], device=x_t.device)
        logits = self.forward(input_ids_q, x_t, timestep)
        return torch.sigmoid(logits)

    def get_reward_gradient(
        self,
        input_ids_q: torch.Tensor,
        x_t: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        lrm_scale: float = 1.0,
        **_,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if timestep is None:
            timestep = torch.zeros(x_t.shape[0], device=x_t.device)
        was_training = self.training
        self.eval()
        x = x_t.detach().float().requires_grad_(True)
        reward = self.get_reward(input_ids_q, x, timestep)
        objective = torch.log(reward.clamp_min(self.config.reward_eps)).sum() * lrm_scale
        grad = torch.autograd.grad(objective, x, retain_graph=False, create_graph=False)[0]
        if was_training:
            self.train()
        return reward.detach(), grad.detach().to(dtype=x_t.dtype)

    def save_pretrained(self, path: str) -> None:
        import os

        os.makedirs(path, exist_ok=True)
        torch.save(
            {"config": asdict(self.config), "state_dict": self.state_dict()},
            os.path.join(path, "scorer.pt"),
        )

    @classmethod
    def from_pretrained(cls, path: str, map_location: Optional[str] = None) -> "LatentReasoningScorer":
        import os

        payload: Dict = torch.load(os.path.join(path, "scorer.pt"), map_location=map_location)
        model = cls(LatentScorerConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model
