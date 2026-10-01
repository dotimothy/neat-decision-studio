"""Checkpoint tensors under the names the LLiMa `OnnxBuilder` asks for.

`OnnxBuilder` pulls every weight through `get_param_func(name)` / `check_param_func(name)`.
Laya's checkpoint packs some of them differently, so this source answers a handful of virtual
names by slicing the packed tensor:

    encoder.layers.N.attn.{q,k,v}_proj.weight      <- attn.Wqkv.weight           (3 x 1024 rows)
    encoder.layers.N.mlp.Wi_{in,gate}.weight       <- mlp.Wi.weight              (2 x 2624 rows)
    head.layers.N.self_attn.{q,k,v}_proj.{w,b}     <- self_attn.in_proj_{w,b}
    type_emb.proj.weight                           <- type_emb.weight.T, zero-padded to 16 inputs
    act_head.0h.{weight,bias}                      <- act_head.0 restricted to the pooled columns

ModernBERT norms carry no bias; a zero one is synthesized because ONNX LayerNormalization is
built with an explicit bias input.
"""
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open

_QKV = {"q": 0, "k": 1, "v": 2}


class LayaWeights:
    def __init__(self, model_path: Path, qtype_channels: int = 16):
        self.qtype_channels = qtype_channels
        self._file = safe_open(str(Path(model_path) / "model.safetensors"), "np")
        self._keys = set(self._file.keys())

    def raw(self, name: str) -> np.ndarray:
        return self._file.get_tensor(name).astype(np.float32)

    def _resolve(self, name: str):
        """Returns a thunk producing the tensor, or None if the name is unknown."""
        if name in self._keys:
            return lambda: self.raw(name)

        m = re.fullmatch(r"(encoder\.layers\.\d+\.attn)\.([qkv])_proj\.weight", name)
        if m:
            return lambda: np.split(self.raw(m[1] + ".Wqkv.weight"), 3, axis=0)[_QKV[m[2]]]

        m = re.fullmatch(r"(encoder\.layers\.\d+\.mlp)\.Wi_(in|gate)\.weight", name)
        if m:
            return lambda: np.split(self.raw(m[1] + ".Wi.weight"), 2, axis=0)[m[2] == "gate"]

        m = re.fullmatch(r"(head\.layers\.\d+\.self_attn)\.([qkv])_proj\.(weight|bias)", name)
        if m:
            return lambda: np.split(self.raw(f"{m[1]}.in_proj_{m[3]}"), 3, axis=0)[_QKV[m[2]]]

        if name == "type_emb.proj.weight":
            def type_emb_conv():
                emb = self.raw("type_emb.weight")  # (types, hidden)
                padded = np.zeros((emb.shape[1], self.qtype_channels), dtype=np.float32)
                padded[:, :emb.shape[0]] = emb.T
                return padded
            return type_emb_conv
        if name == "act_head.0h.weight":
            return lambda: self.raw("act_head.0.weight")[:, :self.raw("type_emb.weight").shape[1]]
        if name == "act_head.0h.bias":
            return lambda: self.raw("act_head.0.bias")

        # Bias-free norm: `<norm>.weight` exists and is 1-D, `<norm>.bias` does not.
        if name.endswith(".bias"):
            weight = name[: -len("bias")] + "weight"
            if weight in self._keys and len(self._file.get_slice(weight).get_shape()) == 1:
                return lambda: np.zeros_like(self.raw(weight))
        return None

    def exists(self, name: str) -> bool:
        return self._resolve(name) is not None

    def get(self, name: str) -> np.ndarray:
        thunk = self._resolve(name)
        if thunk is None:
            raise KeyError(f"no tensor named {name!r} in the Laya checkpoint")
        return thunk()
