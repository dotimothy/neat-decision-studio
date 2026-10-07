#!/usr/bin/env python3
"""PyTorch reference for CLM (Contrastive-LM/CLM-v0.1-8B): what its Qwen3-8B encoder embeds
a text to, and what the two heads make of a state and its candidates.

Upstream serves the encoder through vLLM's pooling runner; here it is Hugging Face
transformers in fp32 on the CPU, with the same recipe: the text as the tokenizer gives it, the
last layer's hidden state (after the final norm) at the last token, L2-normalised. The question
layout, the heads and the scoring are upstream's own code, from third_party/CLM. That the
recipe is the right one shows in the model card's own example, which this reproduces
(billing 0.939 for the twice-charged invoice).

    .venv-ref/bin/python tools/clm_reference.py --out build/clm_reference

The dump (cases.json and embeddings.npy) is what the compiled encoder and the board are
compared against (tools/clm_check.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "CLM" / "src"))
from clm.heads import HeadPair                                    # noqa: E402
from clm.schema import answer_from_logits, build_pairs            # noqa: E402

# The first two are the model card's examples; the rest are the kinds of question this
# repository's pages ask.
CASES = [
    {"name": "card: invoice",
     "state": "Customer: my invoice was charged twice and nobody answers the phone!",
     "questions": {
         "urgency": {"type": "noul", "instructions": "Is this urgent?"},
         "department": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
         "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                         "criteria": ["Calm", "Frustrated", "Very angry"]}}},
    {"name": "card: tides", "state": "What causes tides on Earth?",
     "questions": {"rank": {"type": "choice", "instructions": None,
                            "criteria": {"0": "The Moon's gravitational pull.", "1": "Photosynthesis in plants.",
                                         "2": "Because the Earth is round."}}}},
    {"name": "debate: ice", "state": "", "questions": {"q": {"type": "noul", "instructions": "Is ice colder than steam?"}}},
    {"name": "debate: berlin", "state": "", "questions": {"q": {"type": "noul", "instructions": "Is Berlin the capital of Spain?"}}},
    {"name": "debate: square", "state": "", "questions": {"q": {"type": "noul", "instructions": "Can a square have five sides?"}}},
    {"name": "debate: birds", "state": "", "questions": {"q": {"type": "noul", "instructions": "Do birds have wings?"}}},
    {"name": "triage: outage",
     "state": "The checkout page has returned a 500 error for every customer since 09:12.",
     "questions": {
         "page": {"type": "noul", "instructions": "Should the on-call engineer be paged now?"},
         "severity": {"type": "score", "instructions": "How severe is this?",
                      "criteria": ["Cosmetic", "Minor", "Major", "Critical"]},
         "team": {"type": "choice", "instructions": "Which team owns this?",
                  "criteria": {"payments": "Checkout, cards and refunds", "search": "Search and browsing",
                               "accounts": "Sign-in and profiles"}}}},
    {"name": "snake: wall",
     "state": "The snake's head is next to the wall on its left. Food is two squares up. Its body is behind it.",
     "questions": {"move": {"type": "choice", "instructions": "Which way should the snake go?",
                            "criteria": {"up": "Up, toward the food, into an empty square",
                                         "left": "Left, into the wall, which ends the game",
                                         "right": "Right, into an empty square, away from the food"}}}},
    {"name": "chess: mate in one",
     "state": "White to move. White: king g1, rook a1, pawns f2 g2 h2. Black: king g8, pawns f7 g7 h7.",
     "questions": {"move": {"type": "choice", "instructions": "Which move is best for White?",
                            "criteria": {"Ra8": "Rook to a8, checkmate on the back rank",
                                         "Ra7": "Rook to a7, attacking a pawn",
                                         "h3": "Pawn to h3, making room for the king"}}}},
]


class Encoder:
    def __init__(self, path: str, max_tokens: int, dtype: str, weights: str = "checkpoint"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=getattr(torch, dtype)).eval()
        if weights == "int8":
            # What the compiled graphs hold: every matrix in int8, symmetric, one scale per
            # output channel. Activations stay as they are here.
            with torch.no_grad():
                for module in self.model.modules():
                    if isinstance(module, torch.nn.Linear):
                        scale = module.weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
                        module.weight.copy_((module.weight / scale).round().clamp(-127, 127) * scale)
        self.max_tokens = max_tokens
        self.cache: dict[str, tuple[list[int], np.ndarray, np.ndarray]] = {}

    @torch.no_grad()
    def embed(self, text: str) -> tuple[list[int], np.ndarray, np.ndarray]:
        """-> (token ids, the last token's hidden state as it is, the same L2-normalised)."""
        if text not in self.cache:
            ids = self.tokenizer(text)["input_ids"][: self.max_tokens]
            hidden = self.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].float().numpy()
            self.cache[text] = (ids, hidden, hidden / (np.linalg.norm(hidden) + 1e-12))
        return self.cache[text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="models/Qwen3-8B")
    ap.add_argument("--heads", default="models/CLM-v0.1-8B/CLM_v0.1-8B.pt")
    ap.add_argument("--out", default="build/clm_reference")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--weights", default="checkpoint", choices=["checkpoint", "int8"],
                    help="int8: round the encoder's matrices as the compiler does, to see what that costs")
    args = ap.parse_args()

    heads = HeadPair("clm", args.heads, "cpu").ensure()
    print(f"heads: {heads.cfg}, projection {heads.proj_dim}, scale {heads.scale:.3f}, {heads.n_params:,} parameters", flush=True)
    encoder = Encoder(args.encoder, args.max_tokens, args.dtype, args.weights)

    texts, cases = [], []
    for case in CASES:
        start = time.time()
        pairs = build_pairs(case["state"], case["questions"])
        answers, layout = {}, {}
        for qid, (state_text, keys, candidates) in pairs.items():
            for text in [state_text, *candidates]:
                if text not in texts:
                    texts.append(text)
            state = heads.project_states(encoder.embed(state_text)[2][None])
            actions = heads.project_actions(np.stack([encoder.embed(c)[2] for c in candidates]))
            logits = (heads.scale * (actions @ state[0])).tolist()
            answers[qid] = answer_from_logits(case["questions"][qid], keys, logits)
            layout[qid] = {"state": state_text, "keys": keys, "candidates": candidates, "logits": logits}
        cases.append({**case, "layout": layout, "answers": answers})
        print(f"{case['name']} ({time.time() - start:.1f}s)", flush=True)
        for qid, answer in answers.items():
            shown = answer.get("noul", answer.get("choice", answer.get("score")))
            print(f"    {qid}: {shown if isinstance(shown, str) else round(shown, 4)}  "
                  f"{ {k: round(v, 4) for k, v in answer.get('probabilities', {}).items()} or ''}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "hidden.npy", np.stack([encoder.embed(t)[1] for t in texts]).astype(np.float32))
    (out / "cases.json").write_text(json.dumps({
        "encoder": args.encoder, "heads": args.heads, "scale": heads.scale, "head_cfg": heads.cfg,
        "texts": [{"text": t, "ids": encoder.embed(t)[0]} for t in texts], "cases": cases}, indent=1))
    longest = max(len(encoder.embed(t)[0]) for t in texts)
    print(f"{len(texts)} texts, the longest {longest} tokens -> {out}")


if __name__ == "__main__":
    main()
