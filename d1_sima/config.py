"""Static description of a d1 checkpoint, as the graph builders need it."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from sima_lmm.config.vlm_config import BaseConfig


@dataclass
class D1Config(BaseConfig):
    """A d1 model. Both are an LFM2 trunk (pre-norm layers that are either a short convolution or
    grouped-query attention, each followed by a SwiGLU MLP) with a SigLIP2 NaFlex tower and
    LFM2-VL's projector in front of it for pictures.

    d1-omni (`family` "omni") reads its trunk in both directions and has a decision head of
    `nn.TransformerEncoderLayer`s after it. d1-3B (`family` "lm") is LFM2.5-VL-3B as it is, a
    causal language model whose answer is read off its next-token logits.
    Attributes mirror the checkpoint's `config.json`."""
    family: str = "omni"
    causal: bool = False
    # Where the parts are in the checkpoint.
    trunk: str = "encoder"
    tower: str = "vision.tower.vision_model"
    projector: str = "vision.projector"
    hidden_size: int = 1024
    mlp_size: int = 4608
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 64
    num_hidden_layers: int = 16
    layer_types: list = field(default_factory=list)
    conv_taps: int = 3
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    vocab_size: int = 65536
    head_layers: int = 2
    head_heads: int = 16
    # The vision tower and its projector.
    vision_hidden_size: int = 768
    vision_mlp_size: int = 3072
    vision_heads: int = 12
    vision_layers: int = 12
    vision_eps: float = 1e-6
    vision_activation: str = "gelu_pytorch_tanh"
    patch_size: int = 16
    position_grid: int = 16
    projector_hidden_size: int = 2048
    downsample: int = 2

    @classmethod
    def from_checkpoint(cls, model_path: Path) -> "D1Config":
        raw = json.loads((Path(model_path) / "config.json").read_text())
        if raw.get("model_type") not in ("d1_omni", "lfm2_vl"):
            raise ValueError(f"{model_path}: expected a d1_omni or lfm2_vl checkpoint, found {raw.get('model_type')!r}")
        text, vision = raw["text_config"], raw["vision_config"]
        # Not BaseConfig's own loader: its name differs between sima-lmm releases.
        cfg = cls()
        cfg.hidden_size = text["hidden_size"]
        lm = raw["model_type"] == "lfm2_vl"
        if lm:
            cfg.family, cfg.causal = "lm", True
            cfg.trunk, cfg.tower, cfg.projector = "model.language_model", "model.vision_tower.vision_model", "model.multi_modal_projector"
            for unsupported in ("conv_bias",):
                if text.get(unsupported):
                    raise ValueError(f"{model_path}: {unsupported} is set, which this builder does not implement")
            if raw.get("projector_use_layernorm") or not raw.get("projector_bias", True):
                raise ValueError(f"{model_path}: a projector with a layer norm or without biases is not implemented")
            text = {**text, "rope_theta": text["rope_parameters"]["rope_theta"]}
        # LFM2 sizes its MLP in the model code: two thirds of `intermediate_size`, rounded up,
        # unless the checkpoint says its size is already the MLP's.
        if lm and not text.get("block_auto_adjust_ff_dim"):
            cfg.mlp_size = text["intermediate_size"]
        else:
            hidden = int(text["block_ffn_dim_multiplier"] * int(2 * text["intermediate_size"] / 3))
            cfg.mlp_size = text["block_multiple_of"] * ((hidden + text["block_multiple_of"] - 1) // text["block_multiple_of"])
        cfg.num_attention_heads = text["num_attention_heads"]
        cfg.num_key_value_heads = text["num_key_value_heads"]
        cfg.head_dim = text["hidden_size"] // text["num_attention_heads"]
        cfg.num_hidden_layers = text["num_hidden_layers"]
        cfg.layer_types = list(text["layer_types"])
        cfg.conv_taps = text["conv_L_cache"]
        if cfg.conv_taps != 3:
            raise ValueError(f"{model_path}: a {cfg.conv_taps}-tap convolution is not implemented")
        cfg.rope_theta = float(text["rope_theta"])
        cfg.norm_eps = text["norm_eps"]
        cfg.vocab_size = text["vocab_size"]
        cfg.head_layers = 0 if lm else raw["head_layers"]
        cfg.head_heads = text["hidden_size"] // 64
        cfg.downsample = raw.get("downsample_factor", 2)
        cfg.vision_hidden_size = vision["hidden_size"]
        cfg.vision_mlp_size = vision["intermediate_size"]
        cfg.vision_heads = vision["num_attention_heads"]
        cfg.vision_layers = vision["num_hidden_layers"]
        cfg.vision_eps = vision["layer_norm_eps"]
        cfg.vision_activation = vision["hidden_act"]
        cfg.patch_size = vision["patch_size"]
        cfg.position_grid = int(round(vision["num_patches"] ** 0.5))
        cfg.projector_hidden_size = raw["projector_hidden_size"]
        return cfg
