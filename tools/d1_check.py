#!/usr/bin/env python3
"""Compare d1 on the DevKit with the PyTorch reference (tools/d1_reference.py).

The reference cases, with their pictures, are asked of `laya serve` on the board and its
answers set beside the reference's: the decision made, and the largest difference in a
probability. A picture goes as the reference's PNG, so the board reads the very same pixels.

    python3 tools/d1_check.py                         # every case
    python3 tools/d1_check.py --model-dir model-d1    # the model directory under --remote
    python3 tools/d1_check.py --via-app d1            # ask the app's loaded model, over HTTP:
                                                      # nothing more is loaded on the MLA

Needs only ssh access to the board. Before it loads the model it reads how much of the MLA's
memory is held and stops if the model would not fit: a load that fails leaks memory until the
accelerator is reset.
"""
import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path


def ssh(board: str, command: str, stdin: str | None = None) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", board, command], input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"board command failed ({result.returncode}): {command}\n{(result.stdout + result.stderr)[-2000:]}")
    return result.stdout


def probabilities(answer: dict) -> dict:
    return {"no": 1 - answer["noul"], "yes": answer["noul"]} if answer["type"] == "noul" else answer["probabilities"]


def compare(reference: dict, replies: list[dict]):
    same = total = 0
    worst = 0.0
    for case, reply in zip(reference["cases"], replies, strict=True):
        if "error" in reply:
            print(f"{case['name']}: ERROR {reply['error']}")
            total += len(case["questions"])
            continue
        usage = reply["usage"]
        seen = f", picture {usage['image_tokens']} positions in {usage['vision_ms']:.0f} ms" if usage["images"] else ""
        print(f"{case['name']}  ({usage['latency_ms']:.0f} ms: {usage['decisions']} questions in {usage['encoder_passes']} "
              f"passes on the {usage['seq_len']}-token graph, {usage['mla_ms'] - usage['vision_ms']:.0f} ms{seen})")
        for name, ref in case["answers"].items():
            mine, theirs = probabilities(reply["answers"][name]), probabilities(ref)
            best = lambda p: max(p, key=p.get)
            gap = max(abs(mine[k] - theirs[k]) for k in theirs)
            total += 1
            same += best(mine) == best(theirs)
            worst = max(worst, gap)
            print(f"    {'ok ' if best(mine) == best(theirs) else 'DIFFERS'} {name}: {best(mine)} {mine[best(mine)]:.3f} "
                  f"(reference {best(theirs)} {theirs[best(theirs)]:.3f})")
    print(f"answers: {same}/{total} the same as the reference; largest difference in a probability {worst:.3f}")


def via_app(args):
    import urllib.error
    import urllib.request
    url = args.url or f"http://{args.board.split('@')[-1]}:8095"
    reference = json.loads((args.reference / "cases.json").read_text())
    replies = []
    for case in reference["cases"]:
        request = {"model": args.via_app, "state": case["state"], "questions": case["questions"]}
        if case.get("image"):
            request["images"] = [base64.b64encode((args.images / f"{case['image']}.png").read_bytes()).decode()]
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{url}/api/predict", json.dumps(request).encode(),
                                                               {"Content-Type": "application/json"}), timeout=120) as response:
                replies.append(json.load(response))
        except urllib.error.HTTPError as error:
            replies.append(json.load(error))
    compare(reference, replies)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=Path("build/d1_omni_reference"))
    ap.add_argument("--images", type=Path, default=Path("build/d1_omni_reference/images"), help="the reference pictures")
    ap.add_argument("--board", default="sima@192.168.91.225")
    ap.add_argument("--remote", default="/media/nvme/laya")
    ap.add_argument("--model-dir", default="model-d1", help="model directory under --remote")
    ap.add_argument("--app", default="http://127.0.0.1:8095", help="the app on the board, asked how much MLA memory is held")
    ap.add_argument("--seq-lens", default="", help="load only these chains on the board, e.g. 128 (default: all)")
    ap.add_argument("--via-app", metavar="NAME", help="ask this model of the running app (it must be loaded) instead "
                                                      "of starting a runtime: a second copy of a large model need not fit")
    ap.add_argument("--url", default="", help="the app as this machine reaches it (default: the board's address, port 8095)")
    args = ap.parse_args()
    if args.via_app:
        return via_app(args)

    model = f"{args.remote}/{args.model_dir}"
    need = int(ssh(args.board, f"cat {model}/*_stage1_mla.elf | wc -c"))
    held = ssh(args.board, f"curl -s -m 5 {args.app}/api/mla || true")
    try:
        mla = json.loads(held)
        free = mla["total_bytes"] - mla["held_bytes"]
        print(f"MLA: {mla['held_bytes'] / 2**30:.2f} GB held of {mla['total_bytes'] / 2**30:.0f}; the model's graphs are {need / 2**30:.2f} GB")
        if need > free:
            sys.exit("it would not fit beside what is loaded: not loading it")
    except (ValueError, KeyError, TypeError):
        print("could not read the MLA's held memory from the app; loading without that check", file=sys.stderr)

    reference = json.loads((args.reference / "cases.json").read_text())
    requests = ""
    for case in reference["cases"]:
        request = {"state": case["state"], "questions": case["questions"]}
        if case.get("image"):
            request["images"] = [base64.b64encode((args.images / f"{case['image']}.png").read_bytes()).decode()]
        requests += json.dumps(request) + "\n"
    lens = f" --seq-lens {args.seq_lens}" if args.seq_lens else ""
    out = ssh(args.board, f"{args.remote}/laya serve {model}{lens}", requests)
    replies = [json.loads(line) for line in out.splitlines() if line.startswith(('{"answers"', '{"error"'))]
    compare(reference, replies)


if __name__ == "__main__":
    main()
