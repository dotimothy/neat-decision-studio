"""Checkpoint tensors under the names the LLiMa `OnnxBuilder` asks for.

Most names are the checkpoint's own. The ones that are not:

    <trunk>.layers.N.self_attn.q_layernorm_scaled.weight
        q_layernorm.weight / sqrt(head_dim): attention divides its logits by sqrt(head_dim), and
        with a norm between the projection and the product that rides on the norm's weight
    <trunk>.layers.N.conv.tap{0,1,2}.weight
        the three taps of the depthwise convolution, each as a kernel of its own (see model.py):
        three wide for d1-omni, whose taps are the position before, the position itself and the
        one after
    <trunk>.layers.N.conv.causal{0,1,2}.weight
        the same for d1-3B, whose taps are two positions back, one back and the position itself:
        five wide, so that the kernel stays centred
    <projector>.linear_1.part{0,1,2,3}.{weight,bias}
        the projector's first layer cut by its input into the four patches of a merged block
        (the bias goes with part 0): d1-3B's is too wide to go to the MLA as one input
    head.head.layers.N.self_attn.{q,k,v}_proj.{weight,bias}
        the thirds of `nn.MultiheadAttention`'s packed in_proj_weight and in_proj_bias
"""
import json
import re
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}


class D1Weights:
    def __init__(self, model_path: Path, head_dim: int = 64):
        self.head_dim = head_dim
        self._tensors: dict[str, tuple[np.memmap, dict]] = {}
        for shard in sorted(Path(model_path).glob("*.safetensors")):
            with open(shard, "rb") as handle:
                size, = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(size))
            data = np.memmap(shard, dtype=np.uint8, mode="r", offset=8 + size)
            for name, entry in header.items():
                if name != "__metadata__":
                    self._tensors[name] = (data, entry)
        if not self._tensors:
            raise FileNotFoundError(f"no safetensors files in {model_path}")

    def raw(self, name: str) -> np.ndarray:
        """The tensor as float32."""
        data, entry = self._tensors[name]
        begin, end = entry["data_offsets"]
        values = np.frombuffer(data[begin:end], dtype=_DTYPES[entry["dtype"]]).reshape(entry["shape"])
        if entry["dtype"] == "BF16":
            return (values.astype(np.uint32) << 16).view(np.float32)
        return values.astype(np.float32)

    def _resolve(self, name: str):
        if name in self._tensors:
            return lambda: self.raw(name)
        m = re.fullmatch(r"(.+\.layers\.\d+\.self_attn)\.q_layernorm_scaled\.weight", name)
        if m and f"{m[1]}.q_layernorm.weight" in self._tensors:
            return lambda: self.raw(f"{m[1]}.q_layernorm.weight") * self.head_dim ** -0.5
        m = re.fullmatch(r"(.+\.layers\.\d+\.conv)\.(tap|causal)(\d)\.weight", name)
        if m and f"{m[1]}.conv.weight" in self._tensors:
            def tap(base=m[1], causal=m[2] == "causal", index=int(m[3])):
                full = self.raw(f"{base}.conv.weight")                 # (channels, 1, taps)
                kernel = np.zeros((full.shape[0], 1, 1, 2 * full.shape[2] - 1 if causal else full.shape[2]), np.float32)
                kernel[:, 0, 0, index] = full[:, 0, index]
                return kernel
            return tap
        m = re.fullmatch(r"(.+\.linear_1)\.part(\d)\.(weight|bias)", name)
        if m and f"{m[1]}.weight" in self._tensors and (m[3] == "weight" or m[2] == "0"):
            def part(base=m[1], index=int(m[2]), kind=m[3]):
                if kind == "bias":
                    return self.raw(f"{base}.bias")
                full = self.raw(f"{base}.weight")                      # (out, 4 * vision hidden)
                size = full.shape[1] // 4
                return np.ascontiguousarray(full[:, index * size:(index + 1) * size])
            return part
        m = re.fullmatch(r"(head\.head\.layers\.\d+\.self_attn)\.([qkv])_proj\.(weight|bias)", name)
        if m and f"{m[1]}.in_proj_{m[3]}" in self._tensors:
            def third(base=m[1], part="qkv".index(m[2]), kind=m[3]):
                packed = self.raw(f"{base}.in_proj_{kind}")
                size = packed.shape[0] // 3
                return packed[part * size:(part + 1) * size]
            return third
        return None

    def exists(self, name: str) -> bool:
        return self._resolve(name) is not None

    def get(self, name: str) -> np.ndarray:
        thunk = self._resolve(name)
        if thunk is None:
            raise KeyError(f"no tensor named {name!r} in the d1 checkpoint")
        return thunk()
