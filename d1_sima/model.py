"""d1-omni's trunk, decision head and vision tower as MLA graphs, built with the LLiMa compiler
framework.

The three models here are `sima_lmm.model.base.BaseModel`s like `clm_sima.model.QwenLayersModel`,
so the compiler passes (ONNX -> quantized Model SDK -> MLA ELF) are inherited; only the graphs
are new. LLiMa's own LFM2 graphs are cut for writing text a token at a time, causally, with a
cache between steps. d1-omni reads its input once, in both directions, so every position goes
through whole layers together and nothing is cached.

All graphs are in LLiMa's token layout, NCHW with one token (or image patch) per W position.

`D1TrunkModel`: a run of the trunk's layers; the graph that holds the last layer also holds the
decision head and its scorer.

    inputs   hidden_in    (1, hidden, 1, S)   embeddings, or the previous graph's output
             mask         (1, S, 1, S)        additive, [key, query]: what a query may attend to
             conv_left    (1, hidden, 1, S)   1 where a position may be read by the one after it
             conv_right   (1, hidden, 1, S)   1 where a position may be read by the one before it
                                              (the same in every channel: the MLA compiler takes
                                              no input of one channel)
             type_add     (1, hidden, 1, S)   last graph: the question-type embedding of each row
             head_mask    (1, S, 1, S)        last graph: the decision head's attention mask
    output   hidden_out   (1, hidden, 1, S)   after this graph's layers, or from the last graph
             scores       (1, 1, 1, S)        the scorer at every position; read at the markers

The masks are what let one pass read several rows, and a picture in front of a row. The trunk
knows a token's place through RoPE in attention, which depends only on how far apart two tokens
are, and through a three-tap convolution, which reads a token's two neighbours. So rows laid
end to end, each attending only to itself and with the convolution's taps cut at its two ends,
come out exactly as they would alone: `d1_sima.hostio.layout` builds those masks. A picture is
a prefix of embeddings in front of its row; prefix positions attend only to the prefix and the
last of them does not read the text on its right, as in the checkpoint's own code.

`D1VisionModel`: the SigLIP2 tower, for up to N patches of one picture.

    inputs   patches      (1, 3 * patch^2, 1, N)     the picture's patches, each flattened
             positions    (1, vision_hidden, 1, N)   the position embeddings, resized to the
                                                     picture's grid on the CPU
             mask         (1, N, 1, N)               additive: the patches in use
    output   hidden       (1, vision_hidden, 1, N)

`D1ProjectorModel`: LFM2-VL's projector after its 2x2 pixel unshuffle (which is a regrouping of
the tower's output, done on the CPU because it depends on the picture's grid).

    input    merged       (1, 4 * vision_hidden, 1, M)
    output   prefix       (1, hidden, 1, M)          the embeddings that go in front of the text

    Where a merged block is wider than the MLA takes as one input (d1-3B's is 4608 channels),
    the projector takes the block's four patches as four inputs, `merged0` to `merged3`, and
    its first layer is cut the same way: the sum of the four parts is the layer.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sima_lmm.model.base import BaseModel, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxBuilder, OnnxNode

from d1_sima.config import D1Config
from d1_sima.weights import D1Weights


@dataclass
class _D1Model(BaseModel):
    """What the three graphs share: the checkpoint and plain masked attention."""
    seq_len: int = field(default=128, kw_only=True)
    model_path: Path = field(default=None, kw_only=True)

    def __post_init__(self):
        assert isinstance(self.cfg, D1Config)
        self._weights = None

    @property
    def weights(self) -> D1Weights:
        if self._weights is None:
            self._weights = D1Weights(self.model_path, self.cfg.head_dim)
        return self._weights

    def check_hf_param(self, name: str) -> bool:
        return self.weights.exists(name)

    def get_hf_param(self, name: str) -> np.ndarray:
        return self.weights.get(name)

    @property
    def _b(self) -> OnnxBuilder:
        return self._onnx_builder

    def _finish(self):
        self._b.create_and_save_model()
        # Set to None to deallocate the memory.
        self._onnx_builder = None
        self._weights = None

    def _build_plain_attention(self, base_name: str, input_node: OnnxNode, mask: OnnxNode, heads: int) -> OnnxNode:
        """Masked multi-head self-attention with biases and no positions, as in SigLIP2 and in
        `nn.MultiheadAttention`. After `build_matmul_and_split_heads` a projection is
        (1, head_dim, heads, S); the 1/sqrt(head_dim) of the logits is folded into the queries."""
        s = self.seq_len
        scale = (self.get_hf_param(f"{base_name}.q_proj.weight").shape[0] // heads) ** -0.5
        q, = self._b.build_matmul_and_split_heads(f"{base_name}.q_proj", input_node, heads, s, post_matmul_scale=scale)
        k, = self._b.build_matmul_and_split_heads(f"{base_name}.k_proj", input_node, heads, s)
        v, = self._b.build_matmul_and_split_heads(f"{base_name}.v_proj", input_node, heads, s)
        return self._attend(base_name, q, k, v, mask, heads, "out_proj")

    def _attend(self, base_name: str, q: OnnxNode, k: OnnxNode, v: OnnxNode, mask: OnnxNode, heads: int, out_proj: str) -> OnnxNode:
        # (1, key, heads, query), softmax over keys.
        weights = self._b.build_op(f"{base_name}.attn_weights", [q, k], "Einsum", equation="nchw,nchq->nqhw")
        weights = self._b.build_op(f"{base_name}.masked_attn_weights", [weights, mask], "Add")
        probs = self._b.build_op(f"{base_name}.softmax", [weights], "Softmax", axis=1)
        out = self._b.build_op(f"{base_name}.attn_output", [probs, v], "Einsum", equation="nchw,nqhc->nqhw")
        return self._b.build_merge_heads_and_matmul(f"{base_name}.{out_proj}", [out], heads)

    # ------------------------------------------------------------- MLA layouts

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """Default DRAM layout (HWC16): one graph's output is the next graph's input as it is."""
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        return {}


@dataclass
class D1TrunkModel(_D1Model):
    """
    Attributes:
        seq_len: Number of token positions the graph is compiled for.
        model_path: d1-omni checkpoint directory (config.json and model.safetensors).
        first_layer: Index of the first trunk layer in this graph.
        num_layers: How many consecutive layers it holds. The graph with the trunk's last layer
            also holds the embedding norm, the decision head and the scorer.
    """
    first_layer: int = field(default=0, kw_only=True)
    num_layers: int = field(default=1, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.first_layer < self.first_layer + self.num_layers <= self.cfg.num_hidden_layers
        assert self.cfg.head_dim == 64 and self.cfg.hidden_size // self.cfg.head_heads == 64

    @property
    def is_last(self) -> bool:
        return self.first_layer + self.num_layers == self.cfg.num_hidden_layers

    @property
    def kinds(self) -> list[str]:
        return self.cfg.layer_types[self.first_layer:self.first_layer + self.num_layers]

    @property
    def input_names(self) -> list[str]:
        """The graph's inputs, in order: a run of layers with no convolution has no taps to cut,
        and one with no attention needs no mask."""
        names = ["hidden_in"]
        if "full_attention" in self.kinds:
            names.append("mask")
        if "conv" in self.kinds:
            names += ["conv_back1", "conv_back2"] if self.cfg.causal else ["conv_left", "conv_right"]
        if self.is_last and self.cfg.head_layers:
            names += ["type_add", "head_mask"]
        return names

    def gen_onnx_files(self):
        cfg, s = self.cfg, self.seq_len
        wide = (1, cfg.hidden_size, 1, s)
        shapes = {"hidden_in": wide, "mask": (1, s, 1, s), "conv_left": wide, "conv_right": wide, "conv_back1": wide,
                  "conv_back2": wide, "type_add": wide, "head_mask": (1, s, 1, s)}
        self.create_onnx_builder()
        for name in self.input_names:
            self._b.create_input_node(name, shapes[name])
        nodes = dict(zip(self.input_names, self._b.input_nodes))
        hidden = nodes["hidden_in"]
        rope_q = rope_k = None
        if "mask" in nodes:
            rope_q = self._rope_tables("rope.q", cfg.num_attention_heads)
            rope_k = self._rope_tables("rope.k", cfg.num_key_value_heads)

        for idx in range(self.first_layer, self.first_layer + self.num_layers):
            base = f"{cfg.trunk}.layers.{idx}"
            x = self._b.build_rms_norm(f"{base}.operator_norm", hidden, cfg.norm_eps, 0.0)
            if cfg.layer_types[idx] == "full_attention":
                x = self._build_attention(f"{base}.self_attn", x, nodes["mask"], rope_q, rope_k)
            elif cfg.causal:
                x = self._build_conv(f"{base}.conv", x, "causal", [nodes["conv_back2"], nodes["conv_back1"], None])
            else:
                x = self._build_conv(f"{base}.conv", x, "tap", [nodes["conv_left"], None, nodes["conv_right"]])
            hidden = self._b.build_op(f"{base}.add1", [hidden, x], "Add")
            mlp_in = self._b.build_rms_norm(f"{base}.ffn_norm", hidden, cfg.norm_eps, 0.0)
            hidden = self._b.build_op(f"{base}.add2", [hidden, self._build_mlp(f"{base}.feed_forward", mlp_in)], "Add")

        if self.is_last:               # the final norm, despite its name
            hidden = self._b.build_rms_norm(f"{cfg.trunk}.embedding_norm", hidden, cfg.norm_eps, 0.0)
        if self.is_last and cfg.head_layers:
            hidden = self._build_head(hidden, nodes["type_add"], nodes["head_mask"])
            scores = self._build_scorer(hidden)
            self._b.create_output_node(self._b.get_node_output_name(scores), (1, 1, 1, s))
        else:
            self._b.create_output_node(self._b.get_node_output_name(hidden), (1, cfg.hidden_size, 1, s))
        self._finish()

    # ---------------------------------------------------------------- the trunk

    def _build_mlp(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        """SwiGLU: w2(silu(w1(x)) * w3(x))."""
        gate = self._b.build_conv(f"{base_name}.w1", input_node)
        up = self._b.build_conv(f"{base_name}.w3", input_node)
        act = self._b.build_activation(f"{base_name}.act", gate, "silu")
        gated = self._b.build_op(f"{base_name}.gate", [act, up], "Mul")
        return self._b.build_conv(f"{base_name}.w2", gated)

    def _build_conv(self, base_name: str, input_node: OnnxNode, kind: str, masks: list[OnnxNode | None]) -> OnnxNode:
        """LFM2's short convolution: `out(C * conv(B * x))` with [B, C, x] = in(input) and a
        three-tap depthwise convolution over the positions. In d1-omni the taps are the position
        before, the position itself and the one after (`kind` "tap"); in d1-3B, which is
        causal, two positions back, one back and the position itself ("causal").

        A tap is cut where it would cross from one row into another, or from a picture's prefix
        into its text. The cuts differ from tap to tap, so each tap is a convolution of its own
        over its own masked copy of the input (`masks`, None for the tap that is never cut).
        """
        cfg = self.cfg
        in_proj = self._b.build_conv(f"{base_name}.in_proj", input_node)
        split = self._b.build_op(
            f"{base_name}.in_proj.split", [in_proj], "Split", axis=1,
            output_names=[f"{base_name}.B", f"{base_name}.C", f"{base_name}.x"]
        )
        bx = self._b.build_op(f"{base_name}.mul_bx", [[split, 0], [split, 2]], "Mul")

        width = 2 * cfg.conv_taps - 1 if kind == "causal" else cfg.conv_taps      # centred either way

        def tap(index: int) -> OnnxNode:
            source = bx if masks[index] is None else self._b.build_op(f"{base_name}.cut{index}", [bx, masks[index]], "Mul")
            weight = self._b.create_initializer(f"{base_name}.{kind}{index}.weight")
            return self._b.build_op(
                f"{base_name}.{kind}{index}", [source, weight], "Conv", dilations=[1, 1], group=cfg.hidden_size,
                kernel_shape=[1, width], pads=[0, width // 2, 0, width // 2], strides=[1, 1]
            )

        conv = self._b.build_op(f"{base_name}.sum1", [tap(0), tap(1)], "Add")
        conv = self._b.build_op(f"{base_name}.sum2", [conv, tap(2)], "Add")
        gated = self._b.build_op(f"{base_name}.gate", [conv, [split, 1]], "Mul")
        return self._b.build_conv(f"{base_name}.out_proj", gated)

    def _build_attention(self, base_name: str, input_node: OnnxNode, mask: OnnxNode,
                         rope_q: tuple[OnnxNode, OnnxNode], rope_k: tuple[OnnxNode, OnnxNode]) -> OnnxNode:
        """Grouped-query attention with RMS-normed, rotated queries and keys, in both directions.

        The per-head norms and RoPE both work on the channel axis of (1, head_dim, heads, S), so
        they apply in this layout as they are. Keys and values come out with the smaller number
        of heads and are repeated to the queries' afterwards.
        """
        cfg, s = self.cfg, self.seq_len
        heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        q, = self._b.build_matmul_and_split_heads(f"{base_name}.q_proj", input_node, heads, s)
        k, = self._b.build_matmul_and_split_heads(f"{base_name}.k_proj", input_node, kv_heads, s)
        v, = self._b.build_matmul_and_split_heads(f"{base_name}.v_proj", input_node, kv_heads, s)
        # The 1/sqrt(head_dim) of the attention logits rides on the query norm's weight.
        q = self._b.build_rms_norm(f"{base_name}.q_layernorm_scaled", q, cfg.norm_eps, 0.0)
        k = self._b.build_rms_norm(f"{base_name}.k_layernorm", k, cfg.norm_eps, 0.0)
        q = self._build_rope(f"{base_name}.q_rope", q, rope_q)
        k = self._build_rope(f"{base_name}.k_rope", k, rope_k)
        shape = (1, cfg.head_dim, heads, s)
        k = self._b.build_split_expand_concat(f"{base_name}.k_repeat", k, kv_heads, heads // kv_heads, 2, 2, shape)
        v = self._b.build_split_expand_concat(f"{base_name}.v_repeat", v, kv_heads, heads // kv_heads, 2, 2, shape)
        return self._attend(base_name, q, k, v, mask, heads, "out_proj")

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

    # ------------------------------------------------------------------ the head

    def _build_head(self, hidden: OnnxNode, type_add: OnnxNode, head_mask: OnnxNode) -> OnnxNode:
        """The question-type embedding of each row, then pre-norm `nn.TransformerEncoderLayer`s
        (relu) over the text positions."""
        hidden = self._b.build_op("head.type_emb.add", [hidden, type_add], "Add")
        for idx in range(self.cfg.head_layers):
            base = f"head.head.layers.{idx}"
            norm1 = self._b.build_layer_norm(f"{base}.norm1", hidden)
            attn = self._build_plain_attention(f"{base}.self_attn", norm1, head_mask, self.cfg.head_heads)
            hidden = self._b.build_op(f"{base}.add1", [hidden, attn], "Add")
            norm2 = self._b.build_layer_norm(f"{base}.norm2", hidden)
            fc1 = self._b.build_conv(f"{base}.linear1", norm2)
            relu = self._b.build_op(f"{base}.relu", [fc1], "Relu")
            fc2 = self._b.build_conv(f"{base}.linear2", relu)
            hidden = self._b.build_op(f"{base}.add2", [hidden, fc2], "Add")
        return hidden

    def _build_scorer(self, hidden: OnnxNode) -> OnnxNode:
        norm = self._b.build_layer_norm("head.scorer.0", hidden)
        fc1 = self._b.build_conv("head.scorer.1", norm)
        act = self._b.build_activation("head.scorer.2", fc1, "gelu")
        return self._b.build_conv("head.scorer.3", act)


@dataclass
class D1VisionModel(_D1Model):
    """The SigLIP2 tower. `seq_len` is the number of patches it is compiled for."""

    def gen_onnx_files(self):
        cfg, n = self.cfg, self.seq_len
        base = cfg.tower
        self.create_onnx_builder()
        self._b.create_input_node("patches", (1, 3 * cfg.patch_size ** 2, 1, n))
        self._b.create_input_node("positions", (1, cfg.vision_hidden_size, 1, n))
        self._b.create_input_node("mask", (1, n, 1, n))
        patches, positions, mask = self._b.input_nodes
        hidden = self._b.build_conv(f"{base}.embeddings.patch_embedding", patches)
        hidden = self._b.build_op(f"{base}.embeddings.add", [hidden, positions], "Add")
        for idx in range(cfg.vision_layers):
            layer = f"{base}.encoder.layers.{idx}"
            norm1 = self._b.build_layer_norm(f"{layer}.layer_norm1", hidden, cfg.vision_eps)
            attn = self._build_plain_attention(f"{layer}.self_attn", norm1, mask, cfg.vision_heads)
            hidden = self._b.build_op(f"{layer}.add1", [hidden, attn], "Add")
            norm2 = self._b.build_layer_norm(f"{layer}.layer_norm2", hidden, cfg.vision_eps)
            fc1 = self._b.build_conv(f"{layer}.mlp.fc1", norm2)
            act = self._b.build_activation(f"{layer}.mlp.act", fc1, cfg.vision_activation)
            fc2 = self._b.build_conv(f"{layer}.mlp.fc2", act)
            hidden = self._b.build_op(f"{layer}.add2", [hidden, fc2], "Add")
        hidden = self._b.build_layer_norm(f"{base}.post_layernorm", hidden, cfg.vision_eps)
        self._b.create_output_node(self._b.get_node_output_name(hidden), (1, cfg.vision_hidden_size, 1, n))
        self._finish()


@dataclass
class D1ProjectorModel(_D1Model):
    """LFM2-VL's projector. `seq_len` is the number of merged patches it is compiled for."""

    @property
    def parts(self) -> int:
        """How many inputs the merged block comes as: one, or its four patches apart."""
        return 4 if self.cfg.vision_hidden_size * self.cfg.downsample ** 2 > 4096 else 1

    @property
    def input_names(self) -> list[str]:
        return ["merged"] if self.parts == 1 else [f"merged{i}" for i in range(self.parts)]

    def gen_onnx_files(self):
        cfg, m = self.cfg, self.seq_len
        self.create_onnx_builder()
        for name in self.input_names:
            self._b.create_input_node(name, (1, cfg.vision_hidden_size * cfg.downsample ** 2 // self.parts, 1, m))
        if self.parts == 1:
            fc1 = self._b.build_conv(f"{cfg.projector}.linear_1", self._b.input_nodes[0])
        else:
            fc1 = None
            for i, node in enumerate(list(self._b.input_nodes)):
                part = self._b.build_conv(f"{cfg.projector}.linear_1.part{i}", node)
                fc1 = part if fc1 is None else self._b.build_op(f"{cfg.projector}.linear_1.sum{i}", [fc1, part], "Add")
        act = self._b.build_activation(f"{cfg.projector}.act", fc1, "gelu")
        fc2 = self._b.build_conv(f"{cfg.projector}.linear_2", act)
        self._b.create_output_node(self._b.get_node_output_name(fc2), (1, cfg.hidden_size, 1, m))
        self._finish()
