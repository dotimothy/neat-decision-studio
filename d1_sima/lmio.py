"""What the compiled d1-3B leaves to the CPU: the prompt, the masks, the readout.

d1-3B is a causal language model (LFM2.5-VL-3B). A question is a chat turn that stops where the
answer would begin,

    <bos><|im_start|>user\\n[picture]state\\n\\n\\nQUESTION:\\nquestion and options<|im_end|>\\n<|im_start|>assistant\\n

and the answer is read off the logits of the token that would come next: a softmax over the
tokens that spell the options (`yes` and `no`, a digit, a letter), and over nothing else.
`runtime/src/d1.cpp` does the same in C++; this is its reference, in numpy only. The wording of
the prompt is the checkpoint's own (its `prompt.py`), since that is what the model was trained
to read.

A picture is `<|image_start|>`, one `<image>` token for each merged patch, `<|image_end|>`;
the projector's embeddings take the `<image>` tokens' places.
"""
import json

import numpy as np

from d1_sima.hostio import MASK_NEG

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
IMAGE, IMAGE_START, IMAGE_END = "<image>", "<|image_start|>", "<|image_end|>"
YES_FORMS, NO_FORMS = ("yes", "Yes", "YES"), ("no", "No", "NO")
LETTERS = [chr(ord("A") + i) for i in range(26)]


# ------------------------------------------------------------------------ the prompt

def state_block(state) -> str:
    """A state as it is written in front of `QUESTION:`: text as it is, anything else as JSON."""
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
    return f"{text}\n\n"


def option_codes(labels: list[str]) -> list[str]:
    """The code each option is answered by: the labels themselves when they already are single
    letters, else A, B, C... The checkpoint's code goes on to two-character codes past 26
    options; here a choice has at most 26."""
    labels = [str(label).strip() for label in labels]
    if labels and all(len(label) == 1 and label.isalpha() for label in labels):
        return labels
    if len(labels) > len(LETTERS):
        raise ValueError("a choice has at most 26 options")
    return LETTERS[:len(labels)]


def question_block(question: dict) -> str:
    kind, criteria, instructions = question["type"], question.get("criteria"), question["instructions"]
    if kind == "choice":
        labels = list(criteria)
        lines = "\n".join(f"{code} {criteria[label] or label.replace('_', ' ')}" for code, label in zip(option_codes(labels), labels))
        return f"{instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
    if kind == "score":
        legend = "\n".join(f"{i} {name}" for i, name in enumerate(criteria))
        return f"{instructions}\n\n{legend}\n\nReply with a single digit 0-{len(criteria) - 1} only."
    extra = f"\nYes: {criteria.get('true')}\nNo: {criteria.get('false')}" if criteria else ""
    return f"{instructions}{extra}\n\nReply with yes or no only."


def encode_row(encode, token_id, bos: int, state, question: dict, picture_tokens: list[int] = ()) -> list[int]:
    """One question over one state as token ids. `picture_tokens`: how many positions each
    picture in front of the state takes.

    The text between two special tokens is tokenized on its own, as a tokenizer does that
    meets special tokens in a text.
    """
    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    ids = [bos, token_id(IM_START)]
    if picture_tokens:
        ids += encode("user\n")
        for count in picture_tokens:
            ids += [token_id(IMAGE_START)] + [token_id(IMAGE)] * count + [token_id(IMAGE_END)]
        ids += encode(body + question_block(question))
    else:
        ids += encode("user\n" + body + question_block(question))
    return ids + [token_id(IM_END)] + encode("\n") + [token_id(IM_START)] + encode("assistant\n")


def readout_table(encode) -> dict:
    """The token ids an answer is read from, for every option there can be: the forms of yes and
    of no, the ten digits, and for each letter the letter and, where it is one token of its
    own, the letter after a space. Only single tokens count."""
    def single(texts) -> list[int]:
        out = []
        for text in texts:
            ids = encode(text)
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
        return out

    table = {"yes": single(YES_FORMS), "no": single(NO_FORMS), "digits": [single([str(i)]) for i in range(10)],
             "letters": {letter: single([letter, f" {letter}"]) for letter in LETTERS + [letter.lower() for letter in LETTERS]}}
    if not table["yes"] or not table["no"] or any(not group for group in table["digits"]) or any(not group for group in table["letters"].values()):
        raise RuntimeError("the tokenizer has no single token for an option's code")
    return table


def readout_ids(table: dict, question: dict) -> list[list[int]]:
    """The token ids to score, one group an option; an option scores its best form."""
    if question["type"] == "noul":
        return [table["yes"], table["no"]]
    if question["type"] == "score":
        return table["digits"][:len(question["criteria"])]
    return [table["letters"][code] for code in option_codes(list(question["criteria"]))]


def answer(question: dict, logits: list[float]) -> dict:
    """A question's answer from the logits of its options' tokens: `logits[i]` is option i's
    best form. No calibration: the checkpoint ships none."""
    z = np.asarray(logits, np.float64)
    p = np.exp(z - z.max())
    p = (p / p.sum()).tolist()
    if question["type"] == "noul":
        return {"type": "noul", "noul": p[0]}                 # the groups are [yes, no]
    best = int(np.argmax(p))
    if question["type"] == "choice":
        names = list(question["criteria"])
        return {"type": "choice", "choice": names[best], "confidence": p[best], "probabilities": dict(zip(names, p))}
    return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": p[best],
            "probabilities": {str(i): v for i, v in enumerate(p)}}


# ------------------------------------------------------------------------- the masks

def layout(lengths: list[int], seq_len: int) -> dict:
    """The causal trunk's masks for rows laid end to end: `mask` (1, key, 1, query), a query
    seeing the keys of its own row at or before it, and `conv_back1`, `conv_back2`
    (1, 1, 1, S), 1 where a position may be read by the one after it and by the one two after
    it, which is to say where those are in its row. The graph takes the last two repeated over
    its channels (`hostio.wide`). Positions past the last row see themselves only."""
    if sum(lengths) > seq_len:
        raise ValueError(f"{sum(lengths)} tokens do not fit {seq_len} positions")
    open_ = np.eye(seq_len, dtype=bool)                                               # [key, query]
    back1, back2 = np.zeros(seq_len, np.float32), np.zeros(seq_len, np.float32)
    starts, at = [], 0
    for length in lengths:
        starts.append(at)
        row = np.arange(at, at + length)
        open_[np.ix_(row, row)] = row[:, None] <= row[None, :]
        back1[row[:-1]] = 1.0
        back2[row[:-2]] = 1.0
        at += length
    return {"starts": starts, "mask": np.where(open_, 0.0, MASK_NEG).astype(np.float32).reshape(1, seq_len, 1, seq_len),
            "conv_back1": back1.reshape(1, 1, 1, seq_len), "conv_back2": back2.reshape(1, 1, 1, seq_len)}


# ----------------------------------------------------------------------- the picture

def _cubic(x: np.ndarray) -> np.ndarray:
    """The bicubic kernel PIL and torchvision resample with (a = -0.5)."""
    x = np.abs(x)
    return np.where(x < 1, (1.5 * x - 2.5) * x * x + 1, np.where(x < 2, ((-0.5 * x + 2.5) * x - 4) * x + 2, 0.0))


def resize_bicubic(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """An (H, W, C) array resampled to (height, width, C) with the bicubic filter, antialiased:
    LFM2-VL's processor resizes a picture this way, where d1-omni's uses the bilinear one."""
    def taps(size_in: int, size_out: int):
        scale = size_in / size_out
        support = 2.0 * max(scale, 1.0)
        out = []
        for i in range(size_out):
            centre = (i + 0.5) * scale
            low, high = max(0, int(centre - support + 0.5)), min(size_in, int(centre + support + 0.5))
            weights = _cubic((np.arange(low, high) - centre + 0.5) / max(scale, 1.0))
            out.append((low, weights / weights.sum()))
        return out

    wide = np.zeros((image.shape[0], width, image.shape[2]), np.float64)
    for x, (low, weights) in enumerate(taps(image.shape[1], width)):
        wide[:, x] = np.tensordot(image[:, low:low + len(weights)].astype(np.float64), weights, axes=([1], [0]))
    final = np.zeros((height, width, image.shape[2]), np.float64)
    for y, (low, weights) in enumerate(taps(image.shape[0], height)):
        final[y] = np.tensordot(wide[low:low + len(weights)], weights, axes=([0], [0]))
    return final

