"""Checkpoint tensors under the names the LLiMa `OnnxBuilder` asks for.

The checkpoint is several safetensors shards of bfloat16, which numpy has no type for, so the
shards are read here directly: a safetensors file is an 8-byte header length, a JSON header and
the tensors' bytes, and a bfloat16 is the upper half of a float32.

One name is virtual. Attention divides its logits by sqrt(head_dim); with Qwen3's query norm in
between that cannot be folded into the query projection as it is for Laya, so it is folded into
the norm's weight instead:

    model.layers.N.self_attn.q_norm_scaled.weight  <- q_norm.weight / sqrt(head_dim)
"""
import json
import re
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}


class QwenWeights:
    def __init__(self, model_path: Path, head_dim: int = 128):
        self.head_dim = head_dim
        self._tensors: dict[str, tuple[np.memmap, dict]] = {}
        for shard in sorted(Path(model_path).glob("*.safetensors")):
            with open(shard, "rb") as handle:
                size, = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(size))
            data = np.memmap(shard, dtype=np.uint8, mode="r", offset=8 + size)
            for name, entry in header.items():
                # A checkpoint of the bare decoder (no language-model head) has no "model." prefix.
                if name != "__metadata__":
                    known = name if name.startswith(("model.", "lm_head.")) else f"model.{name}"
                    self._tensors[known] = (data, entry)
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

    def bf16(self, name: str) -> np.ndarray:
        """The tensor's bfloat16 bits, as they are in the checkpoint."""
        data, entry = self._tensors[name]
        if entry["dtype"] != "BF16":
            raise TypeError(f"{name} is {entry['dtype']}, not BF16")
        begin, end = entry["data_offsets"]
        return np.frombuffer(data[begin:end], dtype=np.uint16).reshape(entry["shape"])

    def _resolve(self, name: str):
        if name in self._tensors:
            return lambda: self.raw(name)
        m = re.fullmatch(r"(model\.layers\.\d+\.self_attn)\.q_norm_scaled\.weight", name)
        if m and f"{m[1]}.q_norm.weight" in self._tensors:
            return lambda: self.raw(f"{m[1]}.q_norm.weight") * self.head_dim ** -0.5
        return None

    def exists(self, name: str) -> bool:
        return self._resolve(name) is not None

    def get(self, name: str) -> np.ndarray:
        thunk = self._resolve(name)
        if thunk is None:
            raise KeyError(f"no tensor named {name!r} in the Qwen3 checkpoint")
        return thunk()
