"""Small student model for UG-Prune-Lite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vlm_pruning.selectors.ug_prune_lite import coordinate_features


@dataclass(frozen=True)
class UGPruneLiteModelConfig:
    visual_dim: int
    query_dim: int
    hidden_dim: int = 256
    coord_dim: int = 64
    dropout: float = 0.1
    grid: int = 24
    use_global_visual: bool = True
    use_fourier_coords: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class UGPruneLiteModel:
    """Factory wrapper so tests can instantiate after importing torch lazily."""

    @staticmethod
    def build(torch: Any, config: UGPruneLiteModelConfig) -> Any:
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config_payload = config.as_dict()
                coord_input_dim = 14 if config.use_fourier_coords else 2
                self.visual_proj = nn.Linear(config.visual_dim, config.hidden_dim)
                self.query_proj = nn.Linear(config.query_dim, config.hidden_dim)
                self.coord_proj = nn.Linear(coord_input_dim, config.coord_dim)
                global_dim = config.hidden_dim if config.use_global_visual else 0
                fused_dim = config.hidden_dim + config.hidden_dim + config.coord_dim + global_dim
                self.mlp = nn.Sequential(
                    nn.Linear(fused_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                )
                self.target_head = nn.Linear(config.hidden_dim, 1)
                self.context_head = nn.Linear(config.hidden_dim, 1)
                self.keep_head = nn.Linear(config.hidden_dim, 1)

            def forward(self, visual_features: Any, query_embedding: Any, coords: Any | None = None) -> dict[str, Any]:
                if visual_features.ndim != 3:
                    raise ValueError(f"Expected visual_features [B,N,D], got {tuple(visual_features.shape)}")
                if query_embedding.ndim != 2:
                    raise ValueError(f"Expected query_embedding [B,D], got {tuple(query_embedding.shape)}")
                batch, tokens, _dim = visual_features.shape
                if coords is None:
                    coords = coordinate_features(
                        torch,
                        grid=config.grid,
                        device=visual_features.device,
                        fourier=config.use_fourier_coords,
                    )
                if coords.ndim == 2:
                    coords = coords.unsqueeze(0).expand(batch, -1, -1)
                visual = self.visual_proj(visual_features.float())
                query = self.query_proj(query_embedding.float()).unsqueeze(1).expand(-1, tokens, -1)
                coord = self.coord_proj(coords.to(device=visual_features.device, dtype=visual.dtype))
                pieces = [visual, query, coord]
                if config.use_global_visual:
                    global_visual = visual.mean(dim=1, keepdim=True).expand(-1, tokens, -1)
                    pieces.append(global_visual)
                hidden = self.mlp(torch.cat(pieces, dim=-1))
                keep_logits = self.keep_head(hidden).squeeze(-1)
                return {
                    "target_logits": self.target_head(hidden).squeeze(-1),
                    "context_logits": self.context_head(hidden).squeeze(-1),
                    "keep_logits": keep_logits,
                    "score_logits": keep_logits,
                }

        return _Model()
