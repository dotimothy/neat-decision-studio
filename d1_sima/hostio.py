"""What the compiled d1-omni leaves to the CPU: the prompt, the masks, a picture's patches.

`runtime/src/d1.cpp` does the same in C++; this is its reference, in numpy only, and what the
host-side graph check (tools/d1_verify_onnx.py) runs. The layout of a question is the one the
checkpoint documents:

    <bos> <state> state <q> instructions <opt> <mask> option_0 </opt> <opt> <mask> option_1 </opt> ... <decide>

The model scores the hidden state at every `<mask>` and the answer is a softmax over a
question's options. A question is one row. Rows are laid end to end in the graph's positions; a
picture is a prefix of embeddings in front of its row.
"""
import json
import math
import re

import numpy as np

# Large enough that exp() underflows to exactly 0, small enough to stay finite in bfloat16
# once an attention logit is added to it (as for Laya, see laya_sima/hostio.py).
MASK_NEG = -30000.0

QTYPES = {"choice": 0, "score": 1, "noul": 2}
DELIMITERS = {"state": "<|reserved_7|>", "q": "<|reserved_8|>", "opt": "<|reserved_9|>",
              "opt_end": "<|reserved_10|>", "decide": "<|reserved_11|>", "marker": "<|mask|>"}
_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

# A picture is read at 16-pixel patches, merged 2x2: sides are multiples of 32 pixels, and a
# crop is between 64 and 256 merged patches.
PATCH, MERGE, MAX_PATCHES = 16, 2, 1024
MIN_PIXELS, MAX_PIXELS = 64 * 1024, 256 * 1024


# ------------------------------------------------------------------------ the prompt

def escape(text: str) -> str:
    """`<|name|>` -> `<¦name¦>`: text from a caller can never spell a delimiter or a marker."""
    return _SPECIAL.sub(r"<¦\1¦>", text)


def serialize(state) -> str:
    if state is None:
        return ""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def option_texts(question: dict, picture: bool) -> list[str]:
    """The options in the model's order. A noul is two options, false then true; after a picture
    they are worded `no` and `yes`, as the picture questions were trained."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        return [k if v is None or v == "" else f"{k}: {_text(v)}" for k, v in criteria.items()]
    if kind == "score":
        return [f"level {i}: {_text(c)}" for i, c in enumerate(criteria)]
    criteria = criteria or ({"false": "no", "true": "yes"} if picture else {})
    false, true = criteria.get("false", criteria.get("no")), criteria.get("true", criteria.get("yes"))
    return ["false: " + (_text(false) if false not in (None, "") else "no, the statement does not hold"),
            "true: " + (_text(true) if true not in (None, "") else "yes, the statement holds")]


def encode_row(encode, token_id, bos: int, state, question: dict, max_len: int, picture: bool = False,
               per_option: int = 24) -> tuple[list[int], list[int]]:
    """One question over one state as token ids, and where each option's marker is.

    `encode(text)` gives a text's token ids with no special tokens; `token_id(token)` one
    token's id. The options get max(96, min(24 an option + 32, half of `max_len`)) tokens,
    shared out evenly, and the state is cut on the right to the room that is left.
    """
    options = option_texts(question, picture)
    budget = max(96, min(len(options) * per_option + 32, max_len // 2))
    each = max(2, (budget - 3 * len(options)) // len(options))
    tail = ([token_id(DELIMITERS["q"])] + encode(escape(str(question["instructions"]))))[:max(16, budget)]
    markers = []
    for text in options:
        markers.append(len(tail) + 1)
        tail += [token_id(DELIMITERS["opt"]), token_id(DELIMITERS["marker"])] + encode(escape(" " + text))[:each] \
            + [token_id(DELIMITERS["opt_end"])]
    tail.append(token_id(DELIMITERS["decide"]))
    room = max(0, max_len - len(tail) - 2)
    head = [token_id(DELIMITERS["state"])] + encode(escape(serialize(state)))[:room]
    ids = ([bos] + head + tail)[:max_len]
    markers = [m + 1 + len(head) for m in markers]
    if markers[-1] >= max_len:
        raise ValueError("the options do not fit in the context")
    return ids, markers


def temperature_key(question: dict) -> str:
    options = 2 if question["type"] == "noul" else len(question["criteria"])
    return f"{question['type']}:" + ("2" if options <= 2 else "3-5" if options <= 5 else "6-10" if options <= 10 else "11+")


def answer(question: dict, scores: np.ndarray, temperatures: dict, picture: bool) -> dict:
    """A question's answer from the scores at its markers. Text questions are calibrated with a
    temperature for their type and number of options; picture questions are not."""
    z = np.asarray(scores, np.float64)
    if not picture:
        z = z / temperatures.get(temperature_key(question), temperatures.get(question["type"], 1.0))
    p = np.exp(z - z.max())
    p = (p / p.sum()).tolist()
    if question["type"] == "noul":
        return {"type": "noul", "noul": p[1]}                 # the model reads a noul as [false, true]
    best = int(np.argmax(p))
    if question["type"] == "choice":
        names = list(question["criteria"])
        return {"type": "choice", "choice": names[best], "confidence": p[best], "probabilities": dict(zip(names, p))}
    return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": p[best],
            "probabilities": {str(i): v for i, v in enumerate(p)}}


# ------------------------------------------------------------------------- the masks

def layout(rows: list[tuple[int, int]], seq_len: int) -> dict:
    """The trunk's and the head's masks for rows laid end to end.

    A row is (prefix, text): the positions a picture's embeddings take in front of it (0 for a
    row with no picture) and its text tokens. Returns each row's first position and the graph
    inputs `mask`, `head_mask` (1, key, 1, query), `conv_left` and `conv_right` (1, 1, 1, S),
    which the graph takes repeated over its channels (`wide`).
    Positions past the last row attend to themselves only; their outputs are never read.
    """
    if sum(p + t for p, t in rows) > seq_len:
        raise ValueError(f"{sum(p + t for p, t in rows)} positions do not fit {seq_len}")
    open_, head_open = np.eye(seq_len, dtype=bool), np.eye(seq_len, dtype=bool)        # [key, query]
    left, right = np.zeros(seq_len, np.float32), np.zeros(seq_len, np.float32)
    starts, at = [], 0
    for prefix, text in rows:
        starts.append(at)
        row = np.arange(at, at + prefix + text)
        media, words = row[:prefix], row[prefix:]
        open_[np.ix_(row, words)] = True            # a text position reads its whole row
        open_[np.ix_(media, media)] = True          # a prefix position reads the prefix only
        head_open[np.ix_(words, words)] = True      # the head runs over the text alone
        left[row[:-1]] = 1.0                        # read by the next position, inside the row
        right[row[1:]] = 1.0                        # read by the one before it, inside the row
        if prefix:
            right[at + prefix] = 0.0                # the prefix's last position never reads the text
        at += prefix + text
    as_mask = lambda allowed: np.where(allowed, 0.0, MASK_NEG).astype(np.float32).reshape(1, seq_len, 1, seq_len)
    return {"starts": starts, "mask": as_mask(open_), "head_mask": as_mask(head_open),
            "conv_left": left.reshape(1, 1, 1, seq_len), "conv_right": right.reshape(1, 1, 1, seq_len)}


# ----------------------------------------------------------------------- the picture

def wide(taps: np.ndarray, channels: int) -> np.ndarray:
    """A (1, 1, 1, S) input repeated over the graph's channels: the MLA compiler takes no input
    of one channel."""
    return np.ascontiguousarray(np.broadcast_to(taps, (1, channels, 1, taps.shape[3])), np.float32)


def crop_size(width: int, height: int) -> tuple[int, int]:
    """(height, width) a picture is resized to: LFM2-VL's smart resize, to multiples of 32 pixels
    holding between 64 and 256 merged patches. Python's `round` here is to the nearest even."""
    step = PATCH * MERGE
    h, w = max(step, round(height / step) * step), max(step, round(width / step) * step)
    if h * w > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h, w = max(step, math.floor(height / beta / step) * step), max(step, math.floor(width / beta / step) * step)
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h, w = math.ceil(height * beta / step) * step, math.ceil(width * beta / step) * step
    return h, w


def is_tiled(width: int, height: int) -> bool:
    """Whether the checkpoint's own code would cut this picture into tiles and add a thumbnail.
    The board reads such a picture as its thumbnail alone."""
    step = PATCH * MERGE
    return max(PATCH, round(height / step) * step) * max(PATCH, round(width / step) * step) > MAX_PIXELS * 2


def _filter(size_in: int, size_out: int) -> list[tuple[int, np.ndarray]]:
    """Bilinear resampling with antialiasing along one axis: for each output sample, the first
    input sample it reads and the weights. A triangle as wide as the scale when shrinking, as
    PIL and `torch.nn.functional.interpolate(..., antialias=True)` use."""
    scale = size_in / size_out
    support = max(scale, 1.0)
    taps = []
    for i in range(size_out):
        centre = (i + 0.5) * scale
        low, high = max(0, int(centre - support + 0.5)), min(size_in, int(centre + support + 0.5))
        weights = np.clip(1.0 - np.abs((np.arange(low, high) - centre + 0.5) / support), 0.0, None)
        taps.append((low, (weights / weights.sum()).astype(np.float64)))
    return taps


def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """An (H, W, C) array resampled to (height, width, C), in floating point."""
    out = np.zeros((image.shape[0], width, image.shape[2]), np.float64)
    for x, (low, weights) in enumerate(_filter(image.shape[1], width)):
        out[:, x] = np.tensordot(image[:, low:low + len(weights)].astype(np.float64), weights, axes=([1], [0]))
    final = np.zeros((height, width, image.shape[2]), np.float64)
    for y, (low, weights) in enumerate(_filter(image.shape[0], height)):
        final[y] = np.tensordot(out[low:low + len(weights)], weights, axes=([0], [0]))
    return final


def patches(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """An (H, W, 3) uint8 picture, already at its crop size, as (patches, 3 * 16 * 16) in
    [-1, 1] and its grid (rows, columns) of patches. A patch is flattened row, column, colour."""
    h, w, _ = rgb.shape
    ph, pw = h // PATCH, w // PATCH
    x = (rgb.astype(np.float32) - 127.5) / 127.5
    return x.reshape(ph, PATCH, pw, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(ph * pw, PATCH * PATCH * 3), (ph, pw)


def read_picture(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """A picture of any size to its patches: resized to its crop size, then cut up."""
    h, w = crop_size(rgb.shape[1], rgb.shape[0])
    resized = np.clip(np.floor(resize(rgb, h, w) + 0.5), 0, 255).astype(np.uint8)
    return patches(resized)


def positions(table: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """The tower's position embeddings, (16 * 16, hidden), resized to a picture's grid."""
    side = int(round(table.shape[0] ** 0.5))
    return resize(table.reshape(side, side, -1), grid[0], grid[1]).reshape(grid[0] * grid[1], -1).astype(np.float32)


def vision_inputs(patch_values: np.ndarray, grid: tuple[int, int], table: np.ndarray, slots: int) -> dict:
    """The vision graph's inputs for one picture."""
    n = patch_values.shape[0]
    if n > slots:
        raise ValueError(f"{n} patches do not fit {slots}")
    pixels = np.zeros((1, patch_values.shape[1], 1, slots), np.float32)
    pixels[0, :, 0, :n] = patch_values.T
    where = np.zeros((1, table.shape[1], 1, slots), np.float32)
    where[0, :, 0, :n] = positions(table, grid).T
    allowed = np.eye(slots, dtype=bool)
    allowed[:n, :n] = True
    mask = np.where(allowed, 0.0, MASK_NEG).astype(np.float32).reshape(1, slots, 1, slots)
    return {"patches": pixels, "positions": where, "mask": mask}


def projector_inputs(merged: np.ndarray, slots: int, names: list[str]) -> dict:
    """The projector graph's inputs for a picture's merged blocks: one input, or, where the
    graph takes a block's four patches apart, one each."""
    feed = np.zeros((1, merged.shape[1], 1, slots), np.float32)
    feed[0, :, 0, :len(merged)] = merged.T
    size = merged.shape[1] // len(names)
    return {name: np.ascontiguousarray(feed[:, i * size:(i + 1) * size]) for i, name in enumerate(names)}


def unshuffle(hidden: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """The tower's output for a picture's patches, (rows * columns, hidden), regrouped 2x2:
    (rows / 2 * columns / 2, 4 * hidden), a block's four patches side by side in the order
    top left, top right, bottom left, bottom right."""
    rows, columns = grid
    x = hidden[:rows * columns].reshape(rows // MERGE, MERGE, columns // MERGE, MERGE, -1)
    return x.transpose(0, 2, 1, 3, 4).reshape(rows // MERGE * (columns // MERGE), -1)
