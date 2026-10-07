"""A run of Qwen3 decoder layers as one MLA graph, built with the LLiMa compiler framework.

`QwenLayersModel` is a `sima_lmm.model.base.BaseModel` like `laya_sima.model.LayaModel`, so the
compiler passes (ONNX -> quantized Model SDK -> MLA ELF) are inherited; only the graph is new.
LLiMa's own language-model graphs are cut for generating text a token at a time, with a
key-value cache between the parts of a layer. This one is for reading a text once: every
position goes through whole layers together, as in an encoder, and nothing is cached.

The graph is in LLiMa's token layout, NCHW with one token per W position:

    inputs   hidden_in   (1, hidden, 1, S)   token embeddings, or the previous graph's output
             mask        (1, S, 1, S)        additive, [key, query]: which positions a query sees
    output   hidden_out  (1, hidden, 1, S)   after this graph's layers; the last graph also
                                             applies the model's final norm

The mask is what lets one pass read several texts. Attention is causal, and Qwen3 knows a
token's place only through RoPE, which turns queries and keys so that their product depends on
how far apart they are and on nothing else. So texts laid end to end in the S positions, each
allowed to see only itself (a block of the causal triangle a text), come out exactly as they
would alone: `clm_sima.hostio.packed_mask` builds that mask. The hidden state CLM pools, a
text's last token's, is read out of the output at that token's position. A question's state
and all its options then cost one pass, not one each.
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
from sima_lmm.model.base import BaseModel, FileGenPrecision, TensorTessellateParameters, _quantization_params
from sima_lmm.model.onnx_builder import OnnxBuilder, OnnxNode

from clm_sima.config import QwenConfig
from clm_sima.weights import QwenWeights


@dataclass
class QwenLayersModel(BaseModel):
    """
    Attributes:
        seq_len: Number of token positions the graph is compiled for.
        model_path: Qwen3 checkpoint directory (config.json and the safetensors shards).
        first_layer: Index of the first decoder layer in this graph.
        num_layers: How many consecutive layers it holds.
    """
    seq_len: int = field(default=128, kw_only=True)
    model_path: Path = field(default=None, kw_only=True)
    first_layer: int = field(default=0, kw_only=True)
    num_layers: int = field(default=1, kw_only=True)
    # How the matrices are stored, where the pass's own precision is not what is wanted: one of
    # the compiler's per-node weight precisions, by name ("BFLOAT_16_INT4_BLOCKED_WEIGHTS" is
    # the 4-bit format LLiMa's own models use: a scale a block of input channels). With one
    # set, compile with --precision BF16; every matrix is then stored that way.
    weight_precision: str = field(default="", kw_only=True)

    def __post_init__(self):
        assert isinstance(self.cfg, QwenConfig)
        assert 0 <= self.first_layer < self.first_layer + self.num_layers <= self.cfg.num_hidden_layers
        self._weights = None

    @property
    def is_last(self) -> bool:
        return self.first_layer + self.num_layers == self.cfg.num_hidden_layers

    # ------------------------------------------------------------------ weights

    @property
    def weights(self) -> QwenWeights:
        if self._weights is None:
            self._weights = QwenWeights(self.model_path, self.cfg.head_dim)
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
        self._b.create_input_node("hidden_in", (1, cfg.hidden_size, 1, s))
        self._b.create_input_node("mask", (1, s, 1, s))
        hidden, mask = self._b.input_nodes
        rope_q = self._rope_tables("rope.q", cfg.num_attention_heads)
        rope_k = self._rope_tables("rope.k", cfg.num_key_value_heads)

        for idx in range(self.first_layer, self.first_layer + self.num_layers):
            hidden = self._build_layer(idx, hidden, mask, rope_q, rope_k)
        if self.is_last:
            hidden = self._b.build_rms_norm("model.norm", hidden, cfg.norm_eps, 0.0)

        self._b.create_output_node(self._b.get_node_output_name(hidden), (1, cfg.hidden_size, 1, s))
        self._b.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None
        self._weights = None

    def _build_layer(self, idx: int, hidden: OnnxNode, mask: OnnxNode,
                     rope_q: tuple[OnnxNode, OnnxNode], rope_k: tuple[OnnxNode, OnnxNode]) -> OnnxNode:
        cfg, base = self.cfg, f"model.layers.{idx}"
        attn_in = self._b.build_rms_norm(f"{base}.input_layernorm", hidden, cfg.norm_eps, 0.0)
        attn = self._build_attention(f"{base}.self_attn", attn_in, mask, rope_q, rope_k)
        hidden = self._b.build_op(f"{base}.add1", [hidden, attn], "Add")
        mlp_in = self._b.build_rms_norm(f"{base}.post_attention_layernorm", hidden, cfg.norm_eps, 0.0)
        mlp = self._build_mlp(f"{base}.mlp", mlp_in)
        return self._b.build_op(f"{base}.add2", [hidden, mlp], "Add")

    def _build_mlp(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        """SwiGLU: down(act(gate(x)) * up(x))."""
        gate = self._b.build_conv(f"{base_name}.gate_proj", input_node)
        up = self._b.build_conv(f"{base_name}.up_proj", input_node)
        act = self._b.build_activation(f"{base_name}.act", gate, self.cfg.hidden_activation)
        gated = self._b.build_op(f"{base_name}.gate", [act, up], "Mul")
        return self._b.build_conv(f"{base_name}.down_proj", gated)

    def _build_attention(self, base_name: str, input_node: OnnxNode, mask: OnnxNode,
                         rope_q: tuple[OnnxNode, OnnxNode], rope_k: tuple[OnnxNode, OnnxNode]) -> OnnxNode:
        """Causal grouped-query attention with RMS-normed, rotated queries and keys.

        After `build_matmul_and_split_heads` a projection is (1, head_dim, heads, S): channel is
        the position inside a head, H is the head. The per-head norms and RoPE both work on the
        channel axis, so they apply in this layout as they are. Keys and values come out with
        the smaller number of heads and are repeated to the queries' afterwards.
        """
        cfg, s = self.cfg, self.seq_len
        heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        q, = self._b.build_matmul_and_split_heads(f"{base_name}.q_proj", input_node, heads, s)
        k, = self._b.build_matmul_and_split_heads(f"{base_name}.k_proj", input_node, kv_heads, s)
        v, = self._b.build_matmul_and_split_heads(f"{base_name}.v_proj", input_node, kv_heads, s)
        # The 1/sqrt(head_dim) of the attention logits rides on the query norm's weight.
        q = self._b.build_rms_norm(f"{base_name}.q_norm_scaled", q, cfg.norm_eps, 0.0)
        k = self._b.build_rms_norm(f"{base_name}.k_norm", k, cfg.norm_eps, 0.0)
        q = self._build_rope(f"{base_name}.q_rope", q, rope_q)
        k = self._build_rope(f"{base_name}.k_rope", k, rope_k)
        shape = (1, cfg.head_dim, heads, s)
        k = self._b.build_split_expand_concat(f"{base_name}.k_repeat", k, kv_heads, heads // kv_heads, 2, 2, shape)
        v = self._b.build_split_expand_concat(f"{base_name}.v_repeat", v, kv_heads, heads // kv_heads, 2, 2, shape)

        # (1, key, heads, query), softmax over keys.
        weights = self._b.build_op(f"{base_name}.attn_weights", [q, k], "Einsum", equation="nchw,nchq->nqhw")
        weights = self._b.build_op(f"{base_name}.masked_attn_weights", [weights, mask], "Add")
        probs = self._b.build_op(f"{base_name}.softmax", [weights], "Softmax", axis=1)
        out = self._b.build_op(f"{base_name}.attn_output", [probs, v], "Einsum", equation="nchw,nqhc->nqhw")
        return self._b.build_merge_heads_and_matmul(f"{base_name}.o_proj", [out], heads)

    def _rope_tables(self, base_name: str, heads: int) -> tuple[OnnxNode, OnnxNode]:
        """cos and signed-sin tables, (1, head_dim, heads, S), shared by the graph's layers."""
        cfg = self.cfg
        inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, cfg.head_dim, 2, dtype=np.float64) / cfg.head_dim))
        angles = np.outer(inv_freq, np.arange(self.seq_len, dtype=np.float64))  # (half, S)
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=0)
        # rotate_half(x) = cat(-x2, x1): the swap is done in the graph, the sign lives here.
        sin = np.concatenate([-np.sin(angles), np.sin(angles)], axis=0)

        def table(name: str, a: np.ndarray) -> OnnxNode:
            a = np.broadcast_to(a[None, :, None, :], (1, cfg.head_dim, heads, self.seq_len))
            return self._b.create_initializer(name, np.ascontiguousarray(a, dtype=np.float32))

        return table(f"{base_name}.cos", cos), table(f"{base_name}.sin", sin)

    def _build_rope(self, base_name: str, input_node: OnnxNode, rope: tuple[OnnxNode, OnnxNode]) -> OnnxNode:
        cos, sin = rope
        halves = self._b.build_op(
            f"{base_name}.split", [input_node], "Split", axis=1,
            output_names=[f"{base_name}.split_output.{i}" for i in range(2)]
        )
        swapped = self._b.build_op(f"{base_name}.swap", [[halves, 1], [halves, 0]], "Concat", axis=1)
        direct = self._b.build_op(f"{base_name}.cos", [input_node, cos], "Mul")
        rotated = self._b.build_op(f"{base_name}.sin", [swapped, sin], "Mul")
        return self._b.build_op(f"{base_name}.add", [direct, rotated], "Add")

    # ----------------------------------------------------------- quantization

    def gen_model_sdk_files(self, layer_cfg, log_level: int):
        """ONNX -> quantized Model SDK file, as `BaseModel` does, or with `weight_precision`
        set, with every matrix in that precision.

        The compiler takes precision overrides per node, by the node names it assigns itself,
        so the net is quantized once to learn those names and then again with the overrides
        (as laya_sima does for its int8 encoder weights).
        """
        if not self.weight_precision:
            return super().gen_model_sdk_files(layer_cfg, log_level)
        if layer_cfg["precision"] != FileGenPrecision.BF16:
            raise ValueError("weight_precision selects the weights per node; use it with --precision BF16")
        precision = QuantizationPrecision[self.weight_precision]

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
        chosen = [name for name in first._net.nodes["MLA_0"].ir.nodes if re.fullmatch(r"MLA_0/conv2d_\d+", name)]
        del first
        model = quantize(params.with_custom_quantization_configs({
            name: {"quantization_precision": precision} for name in chosen
        }))
        print(f"[quantize] {len(chosen)} matrices stored as {precision.name}", flush=True)
        model.save(self.model_name, self.sima_model_sdk_path, include_unquantized_net=False)

    # ------------------------------------------------------------- MLA layouts

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """Default DRAM layout (HWC16): one graph's output is the next graph's input as it is."""
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        return {}
