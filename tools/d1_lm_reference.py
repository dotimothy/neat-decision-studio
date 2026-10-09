"""Reference answers for LiquidAI's d1-3B, from the PyTorch model on the host.

    .venv-ref/bin/python tools/d1_lm_reference.py --model models/d1-3b --out build/d1_3b_reference

The cases and pictures are those of tools/d1_reference.py (run that first: this reads its
pictures). Written: `cases.json`, with for every question the model's answer, the token ids of
the prompt it read and the token ids its answer is read from, one group an option; and, for a
picture, how many positions it took and the patches the processor made of it (`tensors.npz`).

The model is loaded with its own code (`trust_remote_code`), which is LiquidAI's and stays in
the checkpoint directory: nothing of it is in this repository.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_reference import IMAGE_CASES, TEXT_CASES          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/d1-3b")
    ap.add_argument("--out", default="build/d1_3b_reference")
    ap.add_argument("--images", default="build/d1_omni_reference/images")
    args = ap.parse_args()
    from PIL import Image
    from transformers import AutoModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()
    engine = model.engine
    code = sys.modules[type(engine).__module__.rsplit(".", 1)[0] + ".prompt"]
    tok = engine.tokenizer
    cases, tensors = [], {}
    for case in TEXT_CASES + IMAGE_CASES:
        path = Path(args.images) / f"{case['image']}.png" if case.get("image") else None
        if path is not None and not path.exists():
            continue
        images = None if path is None else [Image.open(path).convert("RGB")]
        answered = model.system_one(case["state"], case["questions"], images=images)
        rows = {}
        for name, raw in case["questions"].items():
            question = code.as_question(raw)
            if images:
                pics = [engine._image_markup(1)]
                text = code.prefix_text(tok, case["state"], engine.bos, engine.state_style, engine.system, pics[0]) \
                    + code.suffix_text(tok, question, engine.lead, engine.option_style)
                inputs = engine._image_inputs(text, [code_cap(engine, image) for image in images])
                ids = inputs["input_ids"][0].tolist()
                key = f"image.{case['name']}"
                if f"{key}.pixel_values" not in tensors:
                    tensors[f"{key}.pixel_values"] = inputs["pixel_values"].float().numpy()
                    tensors[f"{key}.spatial_shapes"] = inputs["spatial_shapes"].numpy()
            else:
                ids = tok.encode(engine.render(case["state"], question), add_special_tokens=False)
            rows[name] = {"ids": ids, "readout": code.readout_ids(tok, question)}
        record = {"name": case["name"], "state": case["state"], "questions": case["questions"], "image": case.get("image"),
                  "answers": answered["answers"], "usage": answered["usage"], "rows": rows}
        if images:
            record["spatial_shapes"] = tensors[f"image.{case['name']}.spatial_shapes"].tolist()
        cases.append(record)
        shown = {n: (round(a["noul"], 4) if a["type"] == "noul" else a.get("choice", round(a.get("score", 0), 3))) for n, a in answered["answers"].items()}
        print(f"  {case['name']:16s} {max(len(r['ids']) for r in rows.values()):4d} tokens  {shown}", flush=True)
    special = {name: tok.convert_tokens_to_ids(name) for name in ("<|im_start|>", "<|im_end|>", "<image>", "<|image_start|>", "<|image_end|>", "<|img_thumbnail|>")}
    (out / "cases.json").write_text(json.dumps({"model": str(args.model), "bos": tok.bos_token, "special": special, "cases": cases}, indent=1))
    np.savez(out / "tensors.npz", **tensors)
    print(f"{len(cases)} cases -> {out}")


def code_cap(engine, image):
    """The picture as the model's own code bounds it before the processor."""
    return sys.modules[type(engine).__module__].cap_pixels(image)


if __name__ == "__main__":
    main()
