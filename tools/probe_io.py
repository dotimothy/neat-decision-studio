#!/usr/bin/env python3
"""Compile tiny graphs to find which I/O shapes the MLA tessellation accepts.

Each probe is a few 1x1 convs with random weights, so quantize + compile takes seconds.
Run under the model-compiler venv:  python tools/probe_io.py [probe ...]
"""
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sima_lmm.config.vlm_config import BaseConfig
from sima_lmm.model.base import BaseModel, FileGenMode, FileGenPrecision, LoraGenMode

S = 128
RNG = np.random.default_rng(0)
i64 = lambda *v: np.array(v, dtype=np.int64)


@dataclass
class Probe(BaseModel):
    kind: str = "base"

    def check_hf_param(self, name):
        return False

    def conv(self, name, node, cin, cout):
        w = self._onnx_builder.create_initializer(
            f"{name}.w", (RNG.standard_normal((cout, cin, 1, 1)) * 0.1).astype(np.float32))
        return self._onnx_builder.build_op(name, [node, w], "Conv")

    def gen_onnx_files(self):
        self.create_onnx_builder()
        b = self._onnx_builder
        out = lambda node, shape: b.create_output_node(b.get_node_output_name(node), shape)
        b.create_input_node("x", (1, 32, 1, S))
        x = b.input_nodes[0]
        h = self.conv("c0", x, 32, 32)
        match self.kind:
            case "base":
                out(h, (1, 32, 1, S))
            case "in_c3":
                b.create_input_node("q", (1, 3, 1, S))
                h = b.build_op("add", [h, self.conv("cq", b.input_nodes[1], 3, 32)], "Add")
                out(h, (1, 32, 1, S))
            case "out_c1":
                out(self.conv("c1", h, 32, 1), (1, 1, 1, S))
            case "out_c16":
                out(self.conv("c1", h, 32, 16), (1, 16, 1, S))
            case "out_w1":
                p = b.build_op("slice", [h, i64(0), i64(1), i64(3)], "Slice")
                out(self.conv("c1", p, 32, 256), (1, 256, 1, 1))
            case "two_out":
                out(self.conv("c1", h, 32, 16), (1, 16, 1, S))
                out(self.conv("c2", h, 32, 32), (1, 32, 1, S))
            case "mask":
                b.create_input_node("m", (1, S, 1, S))
                w = b.build_op("ein", [h, h], "Einsum", equation="nchw,nchq->nqhw")
                w = b.build_op("madd", [w, b.input_nodes[1]], "Add")
                w = b.build_op("sm", [w], "Softmax", axis=1)
                o = b.build_op("ein2", [w, h], "Einsum", equation="nchw,nqhc->nqhw")
                out(o, (1, 32, 1, S))
            case _:
                raise ValueError(self.kind)
        b.create_and_save_model()
        self._onnx_builder = None

    def get_mla_input_tessellate_params(self):
        return {}

    def get_mla_output_tessellate_params(self):
        return {}


def main():
    kinds = sys.argv[1:] or ["base", "in_c3", "out_c1", "out_c16", "out_w1", "two_out", "mask"]
    root = Path("build/probe")
    layer_cfg = {"precision": FileGenPrecision.BF16, "lora": LoraGenMode.LORA_DISABLED}
    results = {}
    for kind in kinds:
        m = Probe(BaseConfig(), f"probe_{kind}", onnx_path=root / "onnx", sima_path=root / "sima", kind=kind)
        try:
            for mode in (FileGenMode.SOURCE_TO_ONNX, FileGenMode.ONNX_TO_QUANT, FileGenMode.MODEL_SDK_COMPILE):
                m.gen_files(mode, layer_cfg=layer_cfg, log_level=logging.ERROR)
            results[kind] = "ok"
        except BaseException as e:  # afe raises SystemExit-like errors too
            tb = traceback.extract_tb(e.__traceback__)[-1]
            results[kind] = f"FAIL {type(e).__name__}: {str(e)[:120]} @ {Path(tb.filename).name}:{tb.name}"
        print(kind, "->", results[kind], flush=True)
    print(results)


if __name__ == "__main__":
    main()
