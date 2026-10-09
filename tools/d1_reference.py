"""Reference answers for LiquidAI's d1-omni, from the PyTorch model on the host.

    .venv-ref/bin/python tools/d1_reference.py --model models/d1-omni-600m --out build/d1_omni_reference

It asks the published checkpoint a fixed set of questions, about text and about pictures, and
writes what the compiled model is then checked against:

    cases.json      every case: its state, its questions, its picture, the model's answers,
                    and for each question the token ids and the option markers' positions
    images/         the pictures, as PNG, so the board is asked about the very same pixels
    tensors.npz     for the graph check (tools/d1_verify_onnx.py): the trunk's and the head's
                    inputs and outputs for some rows, and the vision tower's for a picture

The model is loaded with its own code (`trust_remote_code`), which is LiquidAI's and stays in
the checkpoint directory: nothing of it is in this repository.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

TEXT_CASES = [
    {"name": "refund",
     "state": "I was charged twice this month, please refund one of them.",
     "questions": {
         "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
         "team": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                               "fraud": "Suspected unauthorised use"}},
         "urgency": {"type": "score", "instructions": "How urgent is this?",
                     "criteria": ["Can wait", "Today", "Blocking the customer now"]}}},
    {"name": "outage",
     "state": "The app has been down for our whole team since 9 this morning and we cannot take orders.",
     "questions": {
         "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
         "team": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                               "fraud": "Suspected unauthorised use"}},
         "urgency": {"type": "score", "instructions": "How urgent is this?",
                     "criteria": ["Can wait", "Today", "Blocking the customer now"]}}},
    {"name": "sky", "state": "", "questions": {"q": {"type": "noul", "instructions": "Is the sky blue on a clear day?"}}},
    {"name": "paris", "state": "", "questions": {"q": {"type": "noul", "instructions": "Is Paris the capital of Germany?"}}},
    {"name": "json",
     "state": {"temperature_c": 91, "limit_c": 85, "fan": "on", "load": "high"},
     "questions": {
         "over": {"type": "noul", "instructions": "Is the temperature over its limit?"},
         "action": {"type": "choice", "instructions": "What should the controller do?",
                    "criteria": {"continue": "Keep running as it is", "throttle": "Reduce the load", "shutdown": "Stop now"}}}},
    {"name": "review",
     "state": "The battery lasts two days and the screen is gorgeous, but the camera is mediocre in low light.",
     "questions": {
         "tone": {"type": "choice", "instructions": "What is the overall tone of this review?",
                  "criteria": {"positive": None, "negative": None, "mixed": None}},
         "stars": {"type": "score", "instructions": "How many stars would the reviewer give?",
                   "criteria": ["one star", "two stars", "three stars", "four stars", "five stars"]},
         "camera": {"type": "noul", "instructions": "Does the reviewer praise the camera?"}}},
]


def pictures(out: Path) -> dict:
    """Pictures to ask about: a few drawn here, so that the answers are known, and two photographs
    fetched if the host is online."""
    from PIL import Image, ImageDraw

    made = {}
    image = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(image).ellipse((96, 96, 288, 288), fill=(220, 30, 30))
    made["red_circle"] = image
    image = Image.new("RGB", (640, 320), (20, 20, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 80, 260, 240), fill=(40, 90, 230))
    draw.polygon([(420, 240), (520, 70), (620, 240)], fill=(250, 210, 40))
    made["square_triangle"] = image
    image = Image.new("RGB", (256, 512), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    for i, colour in enumerate([(220, 40, 40), (240, 200, 40), (60, 180, 80)]):
        draw.ellipse((78, 60 + i * 140, 178, 160 + i * 140), fill=colour if i == 2 else (70, 70, 70))
    made["traffic_light"] = image
    for name, url in (("cats", "http://images.cocodataset.org/val2017/000000039769.jpg"),
                      ("street", "http://images.cocodataset.org/val2017/000000087038.jpg")):
        try:
            path = out / f"{name}.jpg"
            if not path.exists():
                path.write_bytes(urllib.request.urlopen(url, timeout=20).read())
            made[name] = Image.open(path).convert("RGB")
        except OSError as error:
            print(f"  no {name} photograph ({error}); carrying on without it", file=sys.stderr)
    for name, image in made.items():
        image.save(out / f"{name}.png")
    return made


IMAGE_CASES = [
    {"name": "red_circle", "image": "red_circle", "state": None, "questions": {
        "circle": {"type": "noul", "instructions": "Is there a red circle in the picture?"},
        "square": {"type": "noul", "instructions": "Is there a blue square in the picture?"},
        "colour": {"type": "choice", "instructions": "What colour is the shape?",
                   "criteria": {"red": None, "green": None, "blue": None, "yellow": None}}}},
    {"name": "square_triangle", "image": "square_triangle", "state": None, "questions": {
        "count": {"type": "score", "instructions": "How many shapes are in the picture?",
                  "criteria": ["none", "one", "two", "three", "four or more"]},
        "triangle": {"type": "noul", "instructions": "Is there a yellow triangle in the picture?"},
        "left": {"type": "choice", "instructions": "Which shape is on the left?",
                 "criteria": {"square": "A blue square", "triangle": "A yellow triangle", "circle": "A red circle"}}}},
    {"name": "traffic_light", "image": "traffic_light", "state": "A driver is approaching this signal.", "questions": {
        "go": {"type": "noul", "instructions": "May the driver go?"},
        "lit": {"type": "choice", "instructions": "Which lamp is lit?",
                "criteria": {"top": "The top lamp", "middle": "The middle lamp", "bottom": "The bottom lamp"}}}},
    {"name": "cats", "image": "cats", "state": None, "questions": {
        "cat": {"type": "noul", "instructions": "Is there a cat in the picture?"},
        "dog": {"type": "noul", "instructions": "Is there a dog in the picture?"},
        "where": {"type": "choice", "instructions": "Where was this picture taken?",
                  "criteria": {"indoors": "Inside a home", "street": "On a street", "nature": "Outdoors in nature"}},
        "count": {"type": "score", "instructions": "How many animals are in the picture?",
                  "criteria": ["none", "one", "two", "three or more"]}}},
    {"name": "street", "image": "street", "state": None, "questions": {
        "people": {"type": "noul", "instructions": "Are there people in the picture?"},
        "scene": {"type": "choice", "instructions": "What kind of scene is this?",
                  "criteria": {"indoors": "Inside a building", "outdoors": "Outside", "studio": "A studio photograph"}}}},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/d1-omni-600m")
    ap.add_argument("--out", default="build/d1_omni_reference")
    args = ap.parse_args()
    from transformers import AutoModel

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()
    code = sys.modules[type(model).__module__]
    tok = model.tokenizer
    made = pictures(out / "images")

    # What goes into and comes out of the trunk, the head and the vision tower, as they run.
    seen = {"trunk": [], "head": [], "tower": [], "projector": []}
    model.encoder.register_forward_hook(lambda m, a, o: seen["trunk"].append((a[0].numpy().copy(), a[1].numpy().copy(), a[2].numpy().copy(), o.numpy().copy())))
    model.head.register_forward_hook(lambda m, a, o: seen["head"].append(tuple(v.numpy().copy() for v in a) + (o.numpy().copy(),)))
    model.vision.tower.register_forward_hook(lambda m, a, k, o: seen["tower"].append(({n: v.numpy().copy() for n, v in k.items()}, o.last_hidden_state.numpy().copy())), with_kwargs=True)
    model.vision.projector.register_forward_hook(lambda m, a, o: seen["projector"].append((a[0].numpy().copy(), o.numpy().copy())))

    cases, tensors = [], {}
    for case in TEXT_CASES + [c for c in IMAGE_CASES if c["image"] in made]:
        for records in seen.values():
            records.clear()
        image = made.get(case.get("image"))
        images = None if image is None else [image]
        answered = model.system_one(case["state"], case["questions"], images=images)
        # The rows as the model's own code lays them out, for the token-level comparison.
        rows = {}
        noul = code.YES_NO if images else None
        prefix = 0 if images is None else int(seen["projector"][-1][1].shape[1]) if len(seen["projector"]) == 1 else sum(p[1].shape[1] for p in seen["projector"])
        limit = min(model.config.image_text_length if images else model.config.max_length, model.config.max_length - prefix)
        state = "" if case["state"] is None else case["state"]
        for name, question in case["questions"].items():
            ids, markers = code.encode(tok, state, code.as_question(question), limit, noul, False)
            rows[name] = {"ids": ids, "markers": markers}
        record = {"name": case["name"], "state": case["state"], "questions": case["questions"], "image": case.get("image"),
                  "answers": answered["answers"], "usage": answered["usage"], "rows": rows, "prefix": prefix}
        if images:
            kwargs, hidden = seen["tower"][0]
            record["spatial_shapes"] = kwargs["spatial_shapes"].tolist()
            record["crops"] = int(kwargs["pixel_values"].shape[0])
            if record["crops"] == 1:                    # one crop: what the board reads
                key = f"image.{case['name']}"
                tensors[f"{key}.pixel_values"] = kwargs["pixel_values"]
                tensors[f"{key}.tower"] = hidden
                tensors[f"{key}.prefix"] = np.concatenate([p[1] for p in seen["projector"]], axis=1)
        if case["name"] in ("refund", "json", "red_circle", "cats", "traffic_light"):
            h, pad, pre, o = seen["trunk"][0]
            key = f"trunk.{case['name']}"
            tensors[f"{key}.in"], tensors[f"{key}.pad"], tensors[f"{key}.prefix"], tensors[f"{key}.out"] = h, pad, pre, o
            text, text_pad, marker_pos, marker_mask, qtype, logits = seen["head"][0]
            for part, value in (("text", text), ("pad", text_pad), ("marker_pos", marker_pos), ("marker_mask", marker_mask),
                                ("qtype", qtype), ("logits", logits)):
                tensors[f"head.{case['name']}.{part}"] = value
        cases.append(record)
        shown = {n: (round(a["noul"], 4) if a["type"] == "noul" else a.get("choice", round(a.get("score", 0), 3))) for n, a in answered["answers"].items()}
        print(f"  {case['name']:16s} prefix {prefix:4d}  {shown}", flush=True)

    (out / "cases.json").write_text(json.dumps({"model": str(args.model), "temperatures": model.config.temperatures, "cases": cases}, indent=1))
    np.savez(out / "tensors.npz", **tensors)
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
