"""Laya as one MLA graph, built with the LLiMa compiler framework.

`LayaModel` is a `sima_lmm.model.base.BaseModel`, the same base the LLiMa Whisper encoder and
vision towers use, so the three compiler passes (ONNX -> quantized Model SDK -> MLA ELF) are
inherited unchanged; only the graph is new.

The graph is in LLiMa's token layout, NCHW with one token per W position:

    inputs   embeds       (1, hidden, 1, S)   token embeddings, before the embedding norm
             global_mask  (1, S, 1, S)        additive, [key, query]: padding
             local_mask   (1, S, 1, S)        additive, [key, query]: padding + sliding window
             qtype        (1, 16, 1, S)       question type, one-hot in the first 3 channels,
                                              repeated per position
    outputs  scores       (1, 1, 1, S)        scorer output at every position
             act_pre      (1, 256, 1, 1)      act head, first layer, pooled-state part only

Three things are arranged differently from the PyTorch model so that nothing is left for the
CPU and nothing needs an operator the MLA lacks:

* RoPE is applied after the heads are split, where `rotate_half` is a swap of two channel
  halves; the sign it carries is folded into the sin table.
* The option gather is turned inside out: the scorer runs at every position and the CPU reads
  the marker positions out of the result.
* The act head's first layer is split by input: the pooled-state columns run here, the four
  probability features (which depend on the gathered logits) are added on the CPU.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx

from afe.apis.defines import gen2_target
from afe.apis.loaded_net import load_model, onnx_source
from afe.core.configs import QuantizationPrecision
from afe.ir.tensor_type import ScalarType
from sima_lmm.model.base import (
    BaseModel, FileGenPrecision, TensorTessellateParameters, _quantization_params
)
from sima_lmm.model.onnx_builder import OnnxBuilder, OnnxNode

from laya_sima.config import LayaConfig
from laya_sima.weights import LayaWeights


@dataclass
class LayaModel(BaseModel):
    """
    Attributes:
        seq_len: Number of token positions the graph is compiled for.
        model_path: Laya checkpoint directory (model.safetensors, encoder/, tokenizer/).
    """
    seq_len: int = field(default=512, kw_only=True)
    model_path: Path = field(default=None, kw_only=True)
    # Diagnostic variant: stop after the encoder and output its hidden state (1, hidden, 1, S).
    # It shows what the MLA's arithmetic does to the features a decision head is trained on.
    encoder_only: bool = field(default=False, kw_only=True)
    # Mixed precision inside the one graph: which encoder weights are stored as int8 while
    # everything else keeps the pass's precision. "mlp" is the three MLP matrices of every
    # encoder layer (two thirds of the weights), "encoder" is every encoder matrix; the decision
    # head is never touched. Latency is mostly weight traffic, so this is the speed knob.
    int8_weights: str = field(default="none", kw_only=True)

    def __post_init__(self):
        assert isinstance(self.cfg, LayaConfig)
        self._weights = None

    # ------------------------------------------------------------------ weights

    @property
    def weights(self) -> LayaWeights:
        if self._weights is None:
            self._weights = LayaWeights(self.model_path, self.cfg.qtype_channels)
        return self._weights

    def check_hf_param(self, name: str) -> bool:
        return self.weights.exists(name)

    def get_hf_param(self, name: str) -> np.ndarray:
        return self.weights.get(name)

    @property
    def _b(self) -> OnnxBuilder:
        return self._onnx_builder

    # -------------------------------------------------------------------- graph

    def gen_onnx_files(self):
        cfg, s = self.cfg, self.seq_len
        self.create_onnx_builder()
        self._b.create_input_node("embeds", (1, cfg.hidden_size, 1, s))
        self._b.create_input_node("global_mask", (1, s, 1, s))
        self._b.create_input_node("local_mask", (1, s, 1, s))
        if self.encoder_only:
            embeds, global_mask, local_mask = self._b.input_nodes
            hidden = self._build_encoder(embeds, global_mask, local_mask)
            self._b.create_output_node(self._b.get_node_output_name(hidden), (1, cfg.hidden_size, 1, s))
            self._b.create_and_save_model()
            self._onnx_builder = None
            self._weights = None
            return
        self._b.create_input_node("qtype", (1, cfg.qtype_channels, 1, s))
        embeds, global_mask, local_mask, qtype = self._b.input_nodes

        hidden = self._build_encoder(embeds, global_mask, local_mask)
        hidden = self._build_head(hidden, global_mask, qtype)
        scores = self._build_scorer(hidden)
        act_pre = self._build_act_pre(hidden)

        self._b.create_output_node(self._b.get_node_output_name(scores), (1, 1, 1, s))
        self._b.create_output_node(
            self._b.get_node_output_name(act_pre), (1, cfg.act_hidden_size, 1, 1)
        )
        self._b.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None
        self._weights = None

    def _build_encoder(self, embeds: OnnxNode, global_mask: OnnxNode, local_mask: OnnxNode) -> OnnxNode:
        cfg = self.cfg
        rope = {
            "full_attention": self._rope_tables("encoder.rope.global", cfg.global_rope_theta),
            "sliding_attention": self._rope_tables("encoder.rope.local", cfg.local_rope_theta),
        }
        masks = {"full_attention": global_mask, "sliding_attention": local_mask}

        hidden = self._b.build_layer_norm("encoder.embeddings.norm", embeds, cfg.norm_eps)
        for idx in range(cfg.num_hidden_layers):
            base = f"encoder.layers.{idx}"
            layer_type = cfg.layer_types[idx]
            # Layer 0 has no attention norm: the embedding norm directly precedes it.
            attn_in = hidden if idx == 0 else self._b.build_layer_norm(
                f"{base}.attn_norm", hidden, cfg.norm_eps
            )
            attn = self._build_attention(
                f"{base}.attn", attn_in, masks[layer_type], out_proj="Wo", rope=rope[layer_type]
            )
            hidden = self._b.build_op(f"{base}.add1", [hidden, attn], "Add")
            mlp_in = self._b.build_layer_norm(f"{base}.mlp_norm", hidden, cfg.norm_eps)
            mlp = self._build_glu_mlp(f"{base}.mlp", mlp_in)
            hidden = self._b.build_op(f"{base}.add2", [hidden, mlp], "Add")
        return self._b.build_layer_norm("encoder.final_norm", hidden, cfg.norm_eps)

    def _build_head(self, hidden: OnnxNode, global_mask: OnnxNode, qtype: OnnxNode) -> OnnxNode:
        """Question-type embedding, then pre-norm `nn.TransformerEncoderLayer`s (relu)."""
        type_emb = self._b.build_conv("type_emb.proj", qtype)
        hidden = self._b.build_op("type_emb.add", [hidden, type_emb], "Add")
        for idx in range(self.cfg.head_layers):
            base = f"head.layers.{idx}"
            norm1 = self._b.build_layer_norm(f"{base}.norm1", hidden)
            attn = self._build_attention(
                f"{base}.self_attn", norm1, global_mask, out_proj="out_proj", rope=None
            )
            hidden = self._b.build_op(f"{base}.add1", [hidden, attn], "Add")
            norm2 = self._b.build_layer_norm(f"{base}.norm2", hidden)
            fc1 = self._b.build_conv(f"{base}.linear1", norm2)
            relu = self._b.build_op(f"{base}.relu", [fc1], "Relu")
            fc2 = self._b.build_conv(f"{base}.linear2", relu)
            hidden = self._b.build_op(f"{base}.add2", [hidden, fc2], "Add")
        return hidden

    def _build_scorer(self, hidden: OnnxNode) -> OnnxNode:
        norm = self._b.build_layer_norm("scorer.0", hidden)
        fc1 = self._b.build_conv("scorer.1", norm)
        act = self._b.build_activation("scorer.2", fc1, "gelu")
        return self._b.build_conv("scorer.3", act)

    def _build_act_pre(self, hidden: OnnxNode) -> OnnxNode:
        i64 = lambda *v: np.array(v, dtype=np.int64)
        pooled = self._b.build_op("act_head.pooled", [hidden, i64(0), i64(1), i64(3)], "Slice")
        return self._b.build_conv("act_head.0h", pooled)

    def _build_glu_mlp(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        """ModernBERT MLP: Wo(gelu(a) * g) with [a, g] = Wi(x). Wi is built as its two halves."""
        act_in = self._b.build_conv(f"{base_name}.Wi_in", input_node)
        gate = self._b.build_conv(f"{base_name}.Wi_gate", input_node)
        act = self._b.build_activation(f"{base_name}.act", act_in, self.cfg.hidden_activation)
        gated = self._b.build_op(f"{base_name}.gate", [act, gate], "Mul")
        return self._b.build_conv(f"{base_name}.Wo", gated)

    def _build_attention(
        self, base_name: str, input_node: OnnxNode, mask_node: OnnxNode, out_proj: str,
        rope: tuple[OnnxNode, OnnxNode] | None
    ) -> OnnxNode:
        """Masked multi-head self-attention, with RoPE on queries and keys if given.

        After `build_matmul_and_split_heads` a projection is (1, head_dim, heads, S): channel is
        the position inside a head, H is the head. That is the layout RoPE wants.
        """
        cfg, s = self.cfg, self.seq_len
        heads = cfg.num_attention_heads
        q, = self._b.build_matmul_and_split_heads(
            f"{base_name}.q_proj", input_node, heads, s, post_matmul_scale=cfg.head_dim ** -0.5
        )
        k, = self._b.build_matmul_and_split_heads(f"{base_name}.k_proj", input_node, heads, s)
        v, = self._b.build_matmul_and_split_heads(f"{base_name}.v_proj", input_node, heads, s)
        if rope is not None:
            q = self._build_rope(f"{base_name}.q_rope", q, rope)
            k = self._build_rope(f"{base_name}.k_rope", k, rope)

        # (1, key, heads, query), softmax over keys.
        weights = self._b.build_op(
            f"{base_name}.attn_weights", [q, k], "Einsum", equation="nchw,nchq->nqhw"
        )
        weights = self._b.build_op(f"{base_name}.masked_attn_weights", [weights, mask_node], "Add")
        probs = self._b.build_op(f"{base_name}.softmax", [weights], "Softmax", axis=1)
        out = self._b.build_op(
            f"{base_name}.attn_output", [probs, v], "Einsum", equation="nchw,nqhc->nqhw"
        )
        return self._b.build_merge_heads_and_matmul(f"{base_name}.{out_proj}", [out], heads)

    def _rope_tables(self, base_name: str, theta: float) -> tuple[OnnxNode, OnnxNode]:
        """cos and signed-sin tables, (1, head_dim, heads, S), shared by all layers of a type."""
        cfg = self.cfg
        half = cfg.head_dim // 2
        inv_freq = 1.0 / (theta ** (np.arange(0, cfg.head_dim, 2, dtype=np.float64) / cfg.head_dim))
        angles = np.outer(inv_freq, np.arange(self.seq_len, dtype=np.float64))  # (half, S)
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=0)
        # rotate_half(x) = cat(-x2, x1): the swap is done in the graph, the sign lives here.
        sin = np.concatenate([-np.sin(angles), np.sin(angles)], axis=0)
        assert cos.shape == (2 * half, self.seq_len)

        def table(name: str, a: np.ndarray) -> OnnxNode:
            a = np.broadcast_to(
                a[None, :, None, :], (1, cfg.head_dim, cfg.num_attention_heads, self.seq_len)
            )
            return self._b.create_initializer(name, np.ascontiguousarray(a, dtype=np.float32))

        return table(f"{base_name}.cos", cos), table(f"{base_name}.sin", sin)

    def _build_rope(self, base_name: str, input_node: OnnxNode, rope: tuple[OnnxNode, OnnxNode]) -> OnnxNode:
        cos, sin = rope
        halves = self._b.build_op(
            f"{base_name}.split", [input_node], "Split", axis=1,
            output_names=[f"{base_name}.split_output.{i}" for i in range(2)]
        )
        swapped = self._b.build_op(
            f"{base_name}.swap", [[halves, 1], [halves, 0]], "Concat", axis=1
        )
        direct = self._b.build_op(f"{base_name}.cos", [input_node, cos], "Mul")
        rotated = self._b.build_op(f"{base_name}.sin", [swapped, sin], "Mul")
        return self._b.build_op(f"{base_name}.add", [direct, rotated], "Add")

    # ----------------------------------------------------------- quantization

    def gen_model_sdk_files(self, layer_cfg, log_level: int):
        """ONNX -> quantized Model SDK file, as `BaseModel` does, plus per-node int8 weights.

        The compiler takes precision overrides per node, by the node names it assigns itself,
        so with `int8_weights` set the net is quantized once to learn those names and then
        again with the overrides.
        """
        if self.int8_weights == "none":
            return super().gen_model_sdk_files(layer_cfg, log_level)
        if layer_cfg["precision"] != FileGenPrecision.BF16:
            raise ValueError("int8_weights selects int8 per node; use it with --precision BF16")

        onnx_model = onnx.load(str(self.onnx_file_name), load_external_data=False)
        shapes = {
            node.name: tuple(d.dim_value for d in node.type.tensor_type.shape.dim)
            for node in onnx_model.graph.input
        }
        del onnx_model
        loaded = load_model(
            onnx_source(str(self.onnx_file_name), shapes, {n: ScalarType.float32 for n in shapes}),
            target=gen2_target, log_level=log_level
        )
        calibration = [{n: np.zeros((s[0], s[2], s[3], s[1]), np.float32) for n, s in shapes.items()}]
        params = _quantization_params(FileGenPrecision.BF16)

        def quantize(quant_params):
            return loaded.quantize(
                calibration_data=calibration, quantization_config=quant_params,
                model_name=self.model_name, log_level=log_level, automatic_layout_conversion=True
            )

        first = quantize(params)
        chosen = self._int8_nodes(first._net.nodes["MLA_0"].ir.nodes)
        del first
        model = quantize(params.with_custom_quantization_configs({
            name: {"quantization_precision": QuantizationPrecision.BFLOAT_16_INT8_WEIGHTS}
            for name in chosen
        }))
        print(f"[quantize] {len(chosen)} convolutions stored with int8 weights", flush=True)
        model.save(self.model_name, self.sima_model_sdk_path, include_unquantized_net=False)

    def _int8_nodes(self, nodes: dict) -> list[str]:
        """Names of the encoder convolutions `int8_weights` selects.

        Encoder matrices are the bias-free convolutions (the compiler names them `conv2d_N`;
        the head's have a bias and are `conv2d_add_N`). The MLP ones are those with the
        intermediate size on one side.
        """
        chosen = []
        for name, node in nodes.items():
            if not re.fullmatch(r"MLA_0/conv2d_\d+", name):
                continue
            attrs = node.ir.attrs
            attrs = attrs if hasattr(attrs, "weight_shape") else attrs.conv_attrs
            shape = tuple(attrs.weight_shape)            # (1, 1, in, 1, out)
            if shape[2] == self.cfg.qtype_channels:
                continue                                 # the question-type embedding
            if self.int8_weights == "encoder" or self.cfg.intermediate_size in shape:
                chosen.append(name)
        return chosen

    # ------------------------------------------------------------- MLA layouts

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """Default DRAM layout (HWC16) for every input: one token per row group."""
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        return {}
