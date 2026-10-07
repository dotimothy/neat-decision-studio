"""Static description of the Qwen3 checkpoint, as the graph builder needs it."""
import json
from dataclasses import dataclass
from pathlib import Path

from sima_lmm.config.vlm_config import BaseConfig


@dataclass
class QwenConfig(BaseConfig):
    """Qwen3 decoder: pre-norm layers of grouped-query attention (RMS-normed queries and keys,
    RoPE) and a SwiGLU MLP. Attributes mirror the checkpoint's `config.json`."""
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    num_hidden_layers: int = 36
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-6
    hidden_activation: str = "silu"
    vocab_size: int = 151936

    @classmethod
    def from_checkpoint(cls, model_path: Path) -> "QwenConfig":
        raw = json.loads((Path(model_path) / "config.json").read_text())
        if raw.get("model_type") != "qwen3":
            raise ValueError(f"{model_path}: expected a qwen3 checkpoint, found {raw.get('model_type')!r}")
        for unsupported in ("attention_bias", "use_sliding_window", "rope_scaling"):
            if raw.get(unsupported):
                raise ValueError(f"{model_path}: {unsupported} is set, which this builder does not implement")
        # Not BaseConfig's own loader: its name differs between sima-lmm releases.
        cfg = cls()
        cfg.hidden_size = raw["hidden_size"]
        cfg.intermediate_size = raw["intermediate_size"]
        cfg.num_attention_heads = raw["num_attention_heads"]
        cfg.num_key_value_heads = raw["num_key_value_heads"]
        cfg.head_dim = raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"])
        cfg.num_hidden_layers = raw["num_hidden_layers"]
        # Newer transformers write the RoPE settings as one `rope_parameters` object.
        rope = raw.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"{model_path}: RoPE type {rope['rope_type']!r} is not implemented")
        cfg.rope_theta = float(raw.get("rope_theta") or rope["rope_theta"])
        cfg.norm_eps = raw["rms_norm_eps"]
        cfg.hidden_activation = raw["hidden_act"]
        cfg.vocab_size = raw["vocab_size"]
        return cfg
