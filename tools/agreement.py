#!/usr/bin/env python3
"""Decision agreement between upstream PyTorch Laya and the compiled model on the DevKit.

Two steps, because they need different environments:

    .venv-ref/bin/python tools/agreement.py reference --out build/agreement.json
    python3 tools/agreement.py board --cases build/agreement.json [--remote /media/nvme/laya]

`reference` answers a fixed grid of states x questions with the fp32 PyTorch model. `board`
sends the same requests through `laya serve` over ssh and reports how often the decision is
the same and how far the probabilities move.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

STATES = [
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    "The app crashes whenever I upload a photo larger than 10 MB. It worked fine before last week's update.",
    "How much does the enterprise plan cost for 200 seats, and is there a discount for annual billing?",
    "Thanks for the quick fix yesterday, everything is working again. Appreciate the help!",
    "URGENT: production database is down, all customers are seeing 500 errors since 09:12 UTC.",
    "I'd like to change the email address on my account from the old company domain to my personal one.",
    "Your service is garbage and so are you. I want my money back right now or I'm calling my lawyer.",
    "Can you remind me how to export my data as CSV? No rush, I just need it sometime this month.",
    "We are evaluating vendors for next quarter and would like a demo of the analytics module.",
    "My invoice shows a charge for a seat we removed in January. Could you correct it when you get a chance?",
    "The password reset email never arrives. I have checked spam. I am locked out and have a client call in an hour.",
    "Subject: Newsletter - 10 tips for a more productive morning. Unsubscribe at any time.",
    "Hey, the dashboard is a little slow today but still usable. Just letting you know.",
    "We need to cancel our subscription effective immediately. The product no longer fits our needs.",
    "Is there an API endpoint to list all users? I could not find it in the documentation.",
    {"ticket_id": 4412, "channel": "chat", "plan": "pro", "message": "Payment failed twice and now my account is suspended"},
    {"sensor": "line-3", "temperature_c": 96, "vibration": "high", "operator_note": "burning smell near motor"},
    "Please delete all my personal data under GDPR. I no longer want an account with you.",
    "Great product! Would love dark mode some day.",
    "I was promised a callback two days ago and nobody called. This is the third time. Extremely disappointed.",
]

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "sales": "pricing, demos, purchases",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "sentiment": {"type": "choice", "instructions": "What is the tone of the message?",
                  "criteria": ["positive", "neutral", "negative"]},
    "needs_human": {"type": "noul", "instructions": "Does this need a human agent rather than an automated reply?"},
}


def decision(answer: dict):
    kind = answer["type"]
    if kind == "choice":
        return answer["choice"]
    if kind == "score":
        probs = answer["probabilities"]
        return max(probs, key=probs.get)
    return answer["noul"] >= 0.5


def probabilities(answer: dict) -> list[float]:
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def cmd_reference(args):
    from laya.agent import Agent
    agent = Agent(args.model, compile=False, device="cpu")
    agent.model.float()
    cases = []
    for state in STATES:
        answers = agent.system_one(state, QUESTIONS)["answers"]
        cases.append({"state": state, "answers": answers})
    Path(args.out).write_text(json.dumps({"questions": QUESTIONS, "cases": cases}, indent=1))
    print(f"wrote {len(cases)} states x {len(QUESTIONS)} questions to {args.out}")


def cmd_board(args):
    data = json.loads(Path(args.cases).read_text())
    requests = "".join(json.dumps({"state": c["state"], "questions": data["questions"]}) + "\n"
                       for c in data["cases"])
    command = f"{args.remote}/laya serve {args.remote}/{args.model_dir}"
    if args.seq_lens:
        command += f" --seq-lens {args.seq_lens}"
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", args.board, command], input=requests,
                            capture_output=True, text=True)
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"answers"')
             or line.startswith('{"error"')]
    if result.returncode != 0 or len(lines) != len(data["cases"]):
        sys.exit(f"board run failed ({result.returncode}), {len(lines)} answers:\n{result.stdout[-1500:]}")

    total = same = 0
    drift, mla_ms, flips = [], [], []
    for case, got in zip(data["cases"], lines):
        if "error" in got:
            sys.exit("board error: " + got["error"])
        mla_ms.append(got["usage"]["mla_ms"] / got["usage"]["decisions"])
        for qid, want in case["answers"].items():
            have = got["answers"][qid]
            total += 1
            delta = max(abs(a - b) for a, b in zip(probabilities(have), probabilities(want)))
            drift.append(delta)
            if decision(have) == decision(want):
                same += 1
            else:
                flips.append(f"  {qid}: torch {decision(want)} ({want['answer_confidence']}) vs "
                             f"board {decision(have)} ({have['answer_confidence']}) | {str(case['state'])[:60]}")
    drift.sort()
    print(f"decisions: {same}/{total} agree with PyTorch ({100 * same / total:.1f}%)")
    print(f"probability drift: mean {sum(drift) / len(drift):.4f}  p95 {drift[int(len(drift) * 0.95)]:.4f}  "
          f"max {drift[-1]:.4f}")
    print(f"MLA per decision: mean {sum(mla_ms) / len(mla_ms):.2f} ms  max {max(mla_ms):.2f} ms")
    if flips:
        print("disagreements:")
        print("\n".join(flips))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--model", default="models/laya")
    ref.add_argument("--out", default="build/agreement.json")
    board = sub.add_parser("board")
    board.add_argument("--cases", default="build/agreement.json")
    board.add_argument("--board", default="sima@192.168.91.225")
    board.add_argument("--remote", default="/media/nvme/laya")
    board.add_argument("--model-dir", default="model", help="model directory under --remote")
    board.add_argument("--seq-lens")
    args = ap.parse_args()
    cmd_reference(args) if args.command == "reference" else cmd_board(args)


if __name__ == "__main__":
    main()
