"""Static description of a Laya checkpoint, as the graph builder needs it."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from sima_lmm.config.vlm_config import BaseConfig

QTYPES = {"choice": 0, "score": 1, "noul": 2}


@dataclass
class LayaConfig(BaseConfig):
    """ModernBERT encoder + Laya decision head.

    Attributes mirror `encoder/config.json` and `rl_agent_config.json` of the checkpoint.
    `sliding_window` is the half-width of local attention: token i sees |i - j| <= sliding_window.
    """
    hidden_size: int = 1024
    intermediate_size: int = 2624
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    layer_types: list[str] = field(default_factory=list)
    local_attention: int = 128
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    hidden_activation: str = "gelu"
    vocab_size: int = 50368
    pad_token_id: int = 50283
    cls_token_id: int = 50281
    sep_token_id: int = 50282
    mask_token_id: int = 50284
    mask_token: str = "[MASK]"

    head_layers: int = 2
    head_intermediate_size: int = 4096
    act_hidden_size: int = 256
    num_qtypes: int = 3
    # The question-type one-hot is padded to a full MLA row: the MLA tessellation rejects a
    # 3-channel input tensor.
    qtype_channels: int = 16
    max_len: int = 512
    head_max_len: int = 192
    temperature: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    temperature_by_options: dict[str, float] = field(default_factory=dict)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def sliding_window(self) -> int:
        return self.local_attention // 2

    @classmethod
    def from_checkpoint(cls, model_path: Path) -> "LayaConfig":
        enc = json.loads((model_path / "encoder" / "config.json").read_text())
        agent = json.loads((model_path / "rl_agent_config.json").read_text())
        cfg = cls()
        # Not BaseConfig's own loader: its name differs between sima-lmm releases.
        for key in ("hidden_size", "intermediate_size", "num_attention_heads",
                    "num_hidden_layers", "local_attention", "norm_eps", "hidden_activation",
                    "vocab_size", "pad_token_id", "cls_token_id", "sep_token_id"):
            setattr(cfg, key, enc[key])
        cfg.layer_types = list(enc["layer_types"])
        rope = enc.get("rope_parameters")
        if isinstance(rope, dict):
            # transformers>=5 layout; 4.x stored global_rope_theta/local_rope_theta directly.
            cfg.global_rope_theta = float(rope["full_attention"]["rope_theta"])
            cfg.local_rope_theta = float(rope["sliding_attention"]["rope_theta"])
        if enc.get("norm_bias") or enc.get("attention_bias") or enc.get("mlp_bias"):
            raise ValueError("encoder biases are not handled by the graph builder")
        if enc.get("hidden_activation", "gelu") != "gelu":
            raise ValueError("only the gelu hidden activation is handled")
        cfg.head_layers = int(agent.get("head_layers", 2))
        cfg.max_len = int(agent.get("max_len", 512))
        cfg.head_max_len = int(agent.get("head_max_len", 192))
        cfg.temperature = [float(t) for t in agent.get("temperature", [1.0, 1.0, 1.0])]
        cfg.temperature_by_options = {
            k: float(v) for k, v in agent.get("temperature_by_options", {}).items()
        }
        if len(agent.get("act_costs", {})) + 1 != 2:
            raise ValueError("only the 2-way act/escalate head is handled")
        # Special tokens come from the tokenizer, not the encoder config: mmBERT's config gives
        # the same id for [CLS] and [SEP], while its tokenizer uses <bos> and <eos>.
        tok = json.loads((model_path / "tokenizer" / "tokenizer.json").read_text())
        tok_cfg = json.loads((model_path / "tokenizer" / "tokenizer_config.json").read_text())
        ids = {t["content"]: t["id"] for t in tok["added_tokens"]}

        def special(name: str) -> tuple[str, int]:
            token = tok_cfg[name]
            token = token["content"] if isinstance(token, dict) else token
            return token, ids[token]

        cfg.mask_token, cfg.mask_token_id = special("mask_token")
        _, cfg.cls_token_id = special("cls_token")
        _, cfg.sep_token_id = special("sep_token")
        _, cfg.pad_token_id = special("pad_token")
        return cfg
