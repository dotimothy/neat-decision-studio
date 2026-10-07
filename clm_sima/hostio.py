"""What the compiled encoder leaves to the CPU that is not a lookup: the attention mask.

`runtime/src/clm.cpp` builds the same mask in C++; this is its reference, in numpy only, and
what the host-side ONNX check runs.
"""
import numpy as np

# Large enough that exp() underflows to exactly 0, small enough to stay finite in bfloat16
# once an attention logit is added to it (as for Laya, see laya_sima/hostio.py).
MASK_NEG = -30000.0


def packed_mask(lengths: list[int], seq_len: int) -> np.ndarray:
    """The mask for texts of these lengths laid end to end, in the graph's (1, key, 1, query).

    A query sees the keys of its own text at or before it. Positions past the last text see
    themselves only; their outputs are never read.
    """
    if sum(lengths) > seq_len:
        raise ValueError(f"{sum(lengths)} tokens do not fit {seq_len} positions")
    open_ = np.eye(seq_len, dtype=bool)
    start = 0
    for length in lengths:
        block = np.arange(start, start + length)
        open_[np.ix_(block, block)] = block[:, None] <= block[None, :]      # [key, query]
        start += length
    return np.where(open_, 0.0, MASK_NEG).astype(np.float32).reshape(1, seq_len, 1, seq_len)


def offsets(lengths: list[int]) -> list[int]:
    """Where each text starts when they are laid end to end."""
    return [int(v) for v in np.concatenate([[0], np.cumsum(lengths)[:-1]])]
