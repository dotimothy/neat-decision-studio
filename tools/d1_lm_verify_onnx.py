#!/usr/bin/env python3
"""Check d1-3B's graphs against the PyTorch model, before anything is quantized.

    .venv-ref/bin/python tools/d1_lm_reference.py --model models/d1-3b --out build/d1_3b_reference
    bin/d1-compile models/d1-3b -o build/d1-3b --layers_per_graph 4 --onnx
    $MODEL_COMPILER_BIN/python tools/d1_lm_verify_onnx.py models/d1-3b --out build/d1-3b --reference build/d1_3b_reference

Every reference case is asked again through the ONNX graphs, in fp32, with everything the board
will do on its CPU done by `d1_sima.lmio` and `d1_sima.hostio`. What is compared: each
question's token ids and the tokens its answer is read from, with the checkpoint's own; a
picture's patches with the processor's; and each question's probabilities, a question alone in
a pass and, for text, a case's questions laid end to end in one.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1_sima import hostio, lmio                              # noqa: E402
from d1_sima.cli import graph_layers, projector_name, trunk_name, vision_name     # noqa: E402
from d1_sima.config import D1Config                           # noqa: E402
from d1_sima.model import D1TrunkModel                        # noqa: E402
from d1_sima.weights import D1Weights                         # noqa: E402


def probabilities(answer: dict) -> list[float]:
    return [answer["noul"], 1 - answer["noul"]] if answer["type"] == "noul" else list(answer["probabilities"].values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_path", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=Path("build/d1_omni_reference/images"))
    ap.add_argument("--seq_lens", default="128,512")
    ap.add_argument("--layers_per_graph", type=int, default=4)
    ap.add_argument("--patches", type=int, default=hostio.MAX_PATCHES)
    ap.add_argument("--cases", default="", help="only these cases, by name")
    args = ap.parse_args()
    import onnxruntime
    from PIL import Image
    from tokenizers import Tokenizer

    cfg = D1Config.from_checkpoint(args.model_path)
    weights = D1Weights(args.model_path, cfg.head_dim)
    raw = json.loads((args.model_path / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.model_path / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    embeddings = weights.raw(f"{cfg.trunk}.embed_tokens.weight")
    table = weights.raw(f"{cfg.tower}.embeddings.position_embedding.weight")
    readout = lmio.readout_table(encode)
    image_id = tokenizer.token_to_id(lmio.IMAGE)
    reference = json.loads((args.reference / "cases.json").read_text())
    tensors = np.load(args.reference / "tensors.npz")
    graphs = graph_layers(cfg, args.layers_per_graph)
    seq_lens = sorted(int(s) for s in args.seq_lens.split(","))
    session = lambda name: onnxruntime.InferenceSession(str(args.out / "onnx_files" / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    names = [D1TrunkModel(cfg, "x", onnx_path=args.out, sima_path=args.out, seq_len=16, model_path=args.model_path,
                          first_layer=first, num_layers=count).input_names for first, count in graphs]
    chains = {}
    tower = projector = None

    def see(path: Path, key: str):
        nonlocal tower, projector
        if tower is None:
            tower, projector = session(vision_name(args.patches)), session(projector_name(args.patches))
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = hostio.crop_size(rgb.shape[1], rgb.shape[0])
        resized = np.clip(np.floor(lmio.resize_bicubic(rgb, h, w) + 0.5), 0, 255).astype(np.uint8)
        values, grid = hostio.patches(resized)
        theirs = tensors[f"{key}.pixel_values"][0, :len(values)]
        off = np.abs(values - theirs) * 127.5
        print(f"      patches {grid}: {(off > 0.5).mean() * 100:.2f}% of values differ, by at most {off.max():.0f} of 255")
        hidden = tower.run(None, hostio.vision_inputs(values, grid, table, args.patches))[0][0, :, 0, :].T
        merged = hostio.unshuffle(hidden, grid)
        feeds = hostio.projector_inputs(merged, args.patches // hostio.MERGE ** 2, [i.name for i in projector.get_inputs()])
        return projector.run(None, feeds)[0][0, :, 0, :len(merged)].T

    def ask(rows: list[tuple[list[int], np.ndarray | None]]) -> list[np.ndarray]:
        """Rows (ids, the picture's embeddings) laid end to end in one pass: each row's final hidden state."""
        lengths = [len(ids) for ids, _ in rows]
        seq_len = next(s for s in seq_lens if s >= sum(lengths))
        if seq_len not in chains:
            chains[seq_len] = [session(trunk_name(seq_len, first, count)) for first, count in graphs]
        plan = lmio.layout(lengths, seq_len)
        hidden = np.zeros((1, cfg.hidden_size, 1, seq_len), np.float32)
        for start, (ids, seen) in zip(plan["starts"], rows):
            row = embeddings[ids].copy()
            if seen is not None:
                row[np.array(ids) == image_id] = seen
            hidden[0, :, 0, start:start + len(ids)] = row.T
        feeds = {"mask": plan["mask"], "conv_back1": hostio.wide(plan["conv_back1"], cfg.hidden_size),
                 "conv_back2": hostio.wide(plan["conv_back2"], cfg.hidden_size)}
        for run, wanted in zip(chains[seq_len], names):
            hidden = run.run(None, {"hidden_in": hidden, **{n: feeds[n] for n in wanted if n != "hidden_in"}})[0]
        return [hidden[0, :, 0, start + length - 1] for start, length in zip(plan["starts"], lengths)]

    worst = {"alone": 0.0, "packed": 0.0}
    flipped = 0
    wanted_cases = set(filter(None, args.cases.split(",")))
    for case in reference["cases"]:
        if wanted_cases and case["name"] not in wanted_cases:
            continue
        print(f"  {case['name']}")
        seen = see(args.images / f"{case['image']}.png", f"image.{case['name']}") if case["image"] else None
        rows = []
        for name, question in case["questions"].items():
            ids = lmio.encode_row(encode, tokenizer.token_to_id, raw["text_config"]["bos_token_id"], case["state"], question,
                                  [len(seen)] if seen is not None else ())
            theirs = case["rows"][name]
            if ids != theirs["ids"]:
                print(f"      {name}: TOKENS DIFFER ({len(ids)} against {len(theirs['ids'])})")
            if lmio.readout_ids(readout, question) != theirs["readout"]:
                print(f"      {name}: READOUT TOKENS DIFFER")
            rows.append((ids, seen))
        passes = {"alone": [ask([row])[0] for row in rows]}
        if seen is None and len(rows) > 1 and sum(len(r[0]) for r in rows) <= seq_lens[-1]:
            passes["packed"] = ask(rows)
        for way, finals in passes.items():
            for (name, question), final in zip(case["questions"].items(), finals):
                logits = [max(float(embeddings[i] @ final) for i in group) for group in lmio.readout_ids(readout, question)]
                mine, theirs = lmio.answer(question, logits), case["answers"][name]
                diff = float(np.abs(np.array(probabilities(mine)) - np.array(probabilities(theirs))).max())
                worst[way] = max(worst[way], diff)
                same = np.argmax(probabilities(mine)) == np.argmax(probabilities(theirs))
                flipped += not same
                print(f"      {way:6s} {name:10s} max probability diff {diff:.5f}{'' if same else '   DECISION DIFFERS'}")
    print(f"worst probability difference: alone {worst['alone']:.5f}, packed {worst['packed']:.5f}; {flipped} decisions differ")
    return 1 if flipped or max(worst.values()) > 0.02 else 0


if __name__ == "__main__":
    sys.exit(main())
