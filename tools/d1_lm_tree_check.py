#!/usr/bin/env python3
"""Check the layout that reads a shared trunk once (`d1_sima.lmio.tree`) against rows read one
by one, with d1-3B's weights in PyTorch on the host.

    .venv-ref/bin/python tools/d1_lm_tree_check.py models/d1-3b --reference build/d1_3b_reference

The language model is run here from its checkpoint as the MLA graphs run it: with the
attention mask, the positions and the convolution's cuts as inputs. With `--reference`, the
text cases of `tools/d1_lm_reference.py` are answered first, to show that this stand-in is the
model. Then a state and several questions are read twice, each question's row alone and all
of them as a trunk with branches, and what is compared is the final hidden state at each
question's last token and the logits read from it: with two ghosts for every convolution
layer, where the two are the same, and with fewer, down to none, which is where
`d1_sima.lmio.GHOSTS` comes from.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1_sima import lmio                                      # noqa: E402
from d1_sima.weights import D1Weights                         # noqa: E402

STATE = ("Ticket 48213, opened Tuesday 09:14 by Dana Whitfield at Northwind Outfitters. Their point-of-sale terminals in "
         "three stores stopped syncing inventory after last night's update, so stock counts on the web shop are wrong and two "
         "customers have already bought items that are not there. Dana says the stores are open and taking orders by hand. "
         "She asks whether the update can be rolled back today, and mentions that their contract renews next month and "
         "that this is the second outage this quarter.")
QUESTIONS = [
    {"type": "noul", "instructions": "Is the customer asking for a rollback?"},
    {"type": "choice", "instructions": "Which team should handle this?",
     "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults", "sales": "Contracts and renewals"}},
    {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "Today", "Blocking the customer now"]},
]


class Trunk:
    """d1-3B's language model as the graphs compute it, in fp32."""

    def __init__(self, model_path: Path):
        raw = json.loads((model_path / "config.json").read_text())["text_config"]
        self.kinds, self.heads, self.kv = raw["layer_types"], raw["num_attention_heads"], raw["num_key_value_heads"]
        self.width, self.eps, self.theta = raw["hidden_size"], raw["norm_eps"], raw["rope_parameters"]["rope_theta"]
        self.dim = self.width // self.heads
        weights = D1Weights(model_path, self.dim)
        self.w = lambda name: torch.from_numpy(weights.raw(f"model.language_model.{name}"))
        self.embeddings = self.w("embed_tokens.weight")

    def norm(self, x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * weight

    @torch.no_grad()
    def __call__(self, hidden: np.ndarray, mask: np.ndarray, back1: np.ndarray, back2: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """`hidden` (S, width); `mask` the graph's (1, key, 1, query); the taps' cuts (S,); a position a slot."""
        h, s = torch.from_numpy(hidden), len(hidden)
        allowed = torch.from_numpy(mask[0, :, 0, :].T.copy())                 # [query, key], additive
        cos, sin = (torch.from_numpy(t[0, :, 0, :].T.copy()) for t in lmio.rope(positions, self.dim, self.theta))   # (S, dim)
        b1, b2 = torch.from_numpy(back1)[:, None], torch.from_numpy(back2)[:, None]
        rotate = lambda x: x * cos[:, None] + torch.cat((x[..., self.dim // 2:], x[..., :self.dim // 2]), -1) * sin[:, None]
        for index, kind in enumerate(self.kinds):
            base = f"layers.{index}"
            x = self.norm(h, self.w(f"{base}.operator_norm.weight"))
            if kind == "conv":
                b, c, u = (x @ self.w(f"{base}.conv.in_proj.weight").T).chunk(3, dim=-1)
                bx, taps = b * u, self.w(f"{base}.conv.conv.weight")[:, 0]     # (width, 3): two back, one back, itself
                zero = torch.zeros(1, self.width)
                one_back = torch.cat((zero, (bx * b1)[:-1]))
                two_back = torch.cat((zero, zero, (bx * b2)[:-2]))
                x = (c * (two_back * taps[:, 0] + one_back * taps[:, 1] + bx * taps[:, 2])) @ self.w(f"{base}.conv.out_proj.weight").T
            else:
                q = (x @ self.w(f"{base}.self_attn.q_proj.weight").T).view(s, self.heads, self.dim)
                k = (x @ self.w(f"{base}.self_attn.k_proj.weight").T).view(s, self.kv, self.dim)
                v = (x @ self.w(f"{base}.self_attn.v_proj.weight").T).view(s, self.kv, self.dim)
                q = rotate(self.norm(q, self.w(f"{base}.self_attn.q_layernorm.weight")))
                k = rotate(self.norm(k, self.w(f"{base}.self_attn.k_layernorm.weight")))
                k, v = (t.repeat_interleave(self.heads // self.kv, dim=1) for t in (k, v))
                scores = torch.einsum("qhd,khd->hqk", q, k) * self.dim ** -0.5 + allowed
                x = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(s, -1) @ self.w(f"{base}.self_attn.out_proj.weight").T
            h = h + x
            y = self.norm(h, self.w(f"{base}.ffn_norm.weight"))
            h = h + (torch.nn.functional.silu(y @ self.w(f"{base}.feed_forward.w1.weight").T) * (y @ self.w(f"{base}.feed_forward.w3.weight").T)) \
                @ self.w(f"{base}.feed_forward.w2.weight").T
        return self.norm(h, self.w("embedding_norm.weight")).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_path", type=Path)
    ap.add_argument("--reference", type=Path, help="the reference's directory: its text cases are answered first")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.model_path / "tokenizer.json"))
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    bos = json.loads((args.model_path / "config.json").read_text())["text_config"]["bos_token_id"]
    trunk = Trunk(args.model_path)
    convs = trunk.kinds.count("conv")
    table = lmio.readout_table(encode)

    def alone(row: list[int]) -> np.ndarray:
        plan = lmio.layout([len(row)], len(row))
        return trunk(trunk.embeddings[row].numpy(), plan["mask"], plan["conv_back1"].ravel(), plan["conv_back2"].ravel(), np.arange(len(row)))[-1]

    def logits(question: dict, final: np.ndarray) -> np.ndarray:
        return np.array([max(float(trunk.embeddings[i].numpy() @ final) for i in group) for group in lmio.readout_ids(table, question)])

    if args.reference:
        worst = 0.0
        for case in json.loads((args.reference / "cases.json").read_text())["cases"]:
            if case["image"]:
                continue
            for name, question in case["questions"].items():
                mine = lmio.answer(question, logits(question, alone(case["rows"][name]["ids"])))
                theirs = case["answers"][name]
                p = lambda a: [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())
                worst = max(worst, float(np.abs(np.array(p(mine)) - np.array(p(theirs))).max()))
        print(f"the reference's text cases, answered by this stand-in: largest difference in a probability {worst:.5f}")

    rows = [lmio.encode_row(encode, tok.token_to_id, bos, STATE, q) for q in QUESTIONS]
    shared = lmio.common(rows)
    print(f"{len(rows)} rows of {[len(r) for r in rows]} tokens; they share their first {shared}; {convs} convolution layers")
    finals = [alone(row) for row in rows]
    wanted = [logits(q, final) for q, final in zip(QUESTIONS, finals)]
    branches = [len(row) - shared for row in rows]
    for ghosts in (2 * convs, 16, 8, 6, 4, 2, 0):
        slots = shared + sum(min(shared, ghosts) + n for n in branches)
        plan = lmio.tree(shared, branches, slots, ghosts)
        hidden = np.zeros((slots, trunk.width), np.float32)
        hidden[:shared] = trunk.embeddings[rows[0][:shared]].numpy()
        ghost = plan["source"] >= 0
        hidden[ghost] = hidden[plan["source"][ghost]]
        for start, row in zip(plan["starts"], rows):
            hidden[start:start + len(row) - shared] = trunk.embeddings[row[shared:]].numpy()
        out = trunk(hidden, plan["mask"], plan["conv_back1"].ravel(), plan["conv_back2"].ravel(), plan["positions"])
        ends = [start + n - 1 for start, n in zip(plan["starts"], branches)]
        state = max(float(np.abs(out[end] - want).max()) for end, want in zip(ends, finals))
        logit = max(float(np.abs(logits(q, out[end]) - want).max()) for q, end, want in zip(QUESTIONS, ends, wanted))
        print(f"  {plan['ghosts']:2d} ghosts a branch, {slots} slots where the rows alone take {sum(map(len, rows))}: "
              f"largest difference in the final state {state:.5f}, in a logit {logit:.5f}")


if __name__ == "__main__":
    main()
