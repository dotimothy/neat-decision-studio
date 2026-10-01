#!/usr/bin/env python3
"""Run upstream Laya (PyTorch, fp32, CPU) on fixed test cases and dump golden tensors.

Runs in `.venv-ref` (torch + transformers + upstream laya), never in the model-compiler venv.
The dump is what every later stage is compared against: the generated ONNX on the host, and
the compiled ELF on the DevKit.

    .venv-ref/bin/python tools/reference.py --model models/laya --out build/reference.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from laya.agent import Agent
from laya.common import build_sequence

CASES = [
    {
        "name": "billing_choice",
        "state": "Hi, we were billed twice for March. Please refund the duplicate today or we "
                 "will cancel our plan.",
        "question": {"type": "choice", "instructions": "Which department should handle this?",
                     "criteria": {"billing": "invoices, payments, refunds",
                                  "technical": "bugs, outages, system errors",
                                  "other": "everything else"}},
    },
    {
        "name": "urgency_score",
        "state": "Hi, we were billed twice for March. Please refund the duplicate today or we "
                 "will cancel our plan.",
        "question": {"type": "score", "instructions": "How urgent is this?",
                     "criteria": ["not urgent", "soon", "blocking"]},
    },
    {
        "name": "churn_noul",
        "state": "Hi, we were billed twice for March. Please refund the duplicate today or we "
                 "will cancel our plan.",
        "question": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    },
    {
        "name": "outage_choice",
        "state": "The dashboard has returned a 502 for every user since the 14:00 deploy. "
                 "Nothing in the release notes mentions the gateway. " * 6,
        "question": {"type": "choice", "instructions": "Which team owns this incident?",
                     "criteria": {"billing": "invoices, payments, refunds",
                                  "platform": "outages, deploys, gateways, infrastructure",
                                  "sales": "pricing, quotes, renewals",
                                  "other": "everything else"}},
    },
]

QTYPE = {"choice": 0, "score": 1, "noul": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/laya")
    ap.add_argument("--out", default="build/reference.npz")
    args = ap.parse_args()

    agent = Agent(args.model, compile=False, device="cpu")
    model = agent.model.float().eval()
    tok = agent.tok
    out = {}
    meta = []
    for case in CASES:
        q = agent._to_internal(case["question"])
        ids, markers = build_sequence(tok, case["state"], q, max_len=agent.cfg["max_len"],
                                      head_max_len=agent.cfg["head_max_len"])
        qtype = QTYPE[q["t"]]
        with torch.no_grad():
            logits, act = model(
                torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long),
                torch.tensor([markers]), torch.ones(1, len(markers), dtype=torch.bool),
                torch.tensor([qtype]))
        n = case["name"]
        out[n + ".ids"] = np.array(ids, dtype=np.int32)
        out[n + ".markers"] = np.array(markers, dtype=np.int32)
        out[n + ".qtype"] = np.array(qtype, dtype=np.int32)
        out[n + ".logits"] = logits[0].numpy()
        out[n + ".act_logits"] = act[0].numpy()
        api = agent.system_one(case["state"], {"q": case["question"]})
        meta.append({"name": n, "state": case["state"], "question": case["question"],
                     "tokens": len(ids), "answer": api["answers"]["q"]})
        print(n, "tokens", len(ids), "logits", logits[0].numpy().round(3),
              "act", act[0].numpy().round(3))
        print("   ", json.dumps(api["answers"]["q"], default=str)[:200])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
