"""Everything the compiled graph leaves to the CPU, in numpy only.

The MLA graph takes token embeddings and returns a score for *every* position plus the first
layer of the act head for the [CLS] position. What stays on the CPU is the part that is either
a table lookup or depends on which positions are option markers:

    before  embedding lookup, the two attention masks, the question-type one-hot
    after   picking marker positions, softmax, the 4 act features and the 256->2 act tail

`runtime/` implements the same contract in C++; this module is its executable specification
and is what the host-side ONNX check runs.
"""
import numpy as np

# Large enough that exp() underflows to exactly 0, small enough to stay finite in bfloat16
# after it is added to an attention logit. A fully masked row (a padded query whose local
# window holds no real token) then softmaxes to a uniform row instead of NaN.
MASK_NEG = -30000.0


def attention_masks(num_tokens: int, seq_len: int, sliding_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Additive masks in the graph's (1, key, 1, query) layout.

    A key is visible if it is a real token; local layers further require |query - key| <=
    sliding_window. Padded queries get the same rule, their outputs are never read.
    """
    key_ok = np.arange(seq_len) < num_tokens
    global_ok = np.broadcast_to(key_ok[:, None], (seq_len, seq_len))
    pos = np.arange(seq_len)
    local_ok = global_ok & (np.abs(pos[:, None] - pos[None, :]) <= sliding_window)
    to_mask = lambda ok: np.where(ok, 0.0, MASK_NEG).astype(np.float32).reshape(1, seq_len, 1, seq_len)
    return to_mask(global_ok), to_mask(local_ok)


def qtype_one_hot(qtype: int, seq_len: int, channels: int = 16) -> np.ndarray:
    one_hot = np.zeros((1, channels, 1, seq_len), dtype=np.float32)
    one_hot[0, qtype] = 1.0
    return one_hot


def embed(ids, table: np.ndarray, seq_len: int, pad_token_id: int) -> np.ndarray:
    """Token embeddings in NCHW (1, hidden, 1, seq_len), padded with the pad token."""
    padded = np.full(seq_len, pad_token_id, dtype=np.int64)
    padded[: len(ids)] = ids
    return np.ascontiguousarray(table[padded].astype(np.float32).T).reshape(1, -1, 1, seq_len)


def gelu(x: np.ndarray) -> np.ndarray:
    from math import erf
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def decode(scores: np.ndarray, act_pre: np.ndarray, markers, act_tail: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-option logits and act logits from the two graph outputs.

    Args:
        scores: per-position scorer output, any shape with seq_len elements.
        act_pre: W_pooled @ h[CLS] + b, any shape with 256 elements.
        markers: positions of the option [MASK] tokens.
        act_tail: `w_feats` (256, 4), `w_out` (2, 256), `b_out` (2,).
    """
    logits = scores.reshape(-1)[np.asarray(markers)].astype(np.float32)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    k = float(max(len(markers), 2))
    entropy = -(p * np.log(np.clip(p, 1e-9, None))).sum() / np.log(k)
    top = np.sort(p)[::-1]
    top1, top2 = top[0], (top[1] if len(top) > 1 else 0.0)
    feats = np.array([top1, top1 - top2, entropy, k / 255.0], dtype=np.float32)
    hidden = gelu(act_pre.reshape(-1).astype(np.float32) + act_tail["w_feats"] @ feats)
    return logits, act_tail["w_out"] @ hidden + act_tail["b_out"]
