#!/usr/bin/env python3
"""Check d1-omni's graphs against the PyTorch model, before anything is quantized.

    .venv-ref/bin/python tools/d1_reference.py --model models/d1-omni-600m --out build/d1_omni_reference
    bin/d1-compile models/d1-omni-600m -o build/d1-omni --onnx
    $MODEL_COMPILER_BIN/python tools/d1_verify_onnx.py models/d1-omni-600m --out build/d1-omni \
        --reference build/d1_omni_reference

Every reference case is asked again through the ONNX graphs, in fp32, with everything the board
will do on its CPU done by `d1_sima.hostio`: the prompt, the masks, a picture's patches and
position embeddings, the regrouping in front of the projector. What is compared:

    tokens      each question's token ids and marker positions, with the checkpoint's own
    pictures    the patches, the tower's output and the prefix, with PyTorch's
    answers     each question's probabilities, a question alone in a pass and, for text, all
                of a case's questions laid end to end in one pass
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d1_sima import hostio                                    # noqa: E402
from d1_sima.cli import graph_layers, projector_name, trunk_name, vision_name     # noqa: E402
from d1_sima.config import D1Config                           # noqa: E402
from d1_sima.model import D1TrunkModel                        # noqa: E402
from d1_sima.weights import D1Weights                         # noqa: E402


def probabilities(answer: dict) -> list[float]:
    return [1 - answer["noul"], answer["noul"]] if answer["type"] == "noul" else list(answer["probabilities"].values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_path", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="d1-compile's output directory")
    ap.add_argument("--reference", type=Path, required=True, help="tools/d1_reference.py's output directory")
    ap.add_argument("--seq_lens", default="128,512")
    ap.add_argument("--layers_per_graph", type=int, default=16)
    ap.add_argument("--patches", type=int, default=hostio.MAX_PATCHES)
    args = ap.parse_args()
    import onnxruntime
    from PIL import Image
    from tokenizers import Tokenizer

    cfg = D1Config.from_checkpoint(args.model_path)
    weights = D1Weights(args.model_path, cfg.head_dim)
    raw = json.loads((args.model_path / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.model_path / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    embeddings = weights.raw("encoder.embed_tokens.weight")
    type_embeddings = weights.raw("head.type_emb.weight")
    table = weights.raw("vision.tower.vision_model.embeddings.position_embedding.weight")
    reference = json.loads((args.reference / "cases.json").read_text())
    tensors = np.load(args.reference / "tensors.npz")
    graphs = graph_layers(cfg, args.layers_per_graph)
    seq_lens = sorted(int(s) for s in args.seq_lens.split(","))
    session = lambda name: onnxruntime.InferenceSession(str(args.out / "onnx_files" / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    chains = {s: [(session(trunk_name(s, first, count)),
                   D1TrunkModel(cfg, "x", onnx_path=args.out, sima_path=args.out, seq_len=s, model_path=args.model_path,
                                first_layer=first, num_layers=count).input_names) for first, count in graphs] for s in seq_lens}
    tower, projector = session(vision_name(args.patches)), session(projector_name(args.patches))

    def see(path: Path, key: str | None):
        """A picture's prefix embeddings, through the vision graphs."""
        values, grid = hostio.read_picture(np.asarray(Image.open(path).convert("RGB")))
        hidden = tower.run(None, hostio.vision_inputs(values, grid, table, args.patches))[0][0, :, 0, :].T
        merged = hostio.unshuffle(hidden, grid)
        slots = args.patches // hostio.MERGE ** 2
        feed = np.zeros((1, merged.shape[1], 1, slots), np.float32)
        feed[0, :, 0, :len(merged)] = merged.T
        prefix = projector.run(None, {"merged": feed})[0][0, :, 0, :len(merged)].T
        if key and f"{key}.pixel_values" in tensors:
            n = len(values)
            theirs = tensors[f"{key}.pixel_values"][0, :n]
            off = np.abs(values - theirs) * 127.5
            print(f"      patches: {(off > 0.5).mean() * 100:.2f}% of values differ, by at most {off.max():.0f} of 255;  "
                  f"tower max diff {np.abs(hidden[:n] - tensors[f'{key}.tower'][0, :n]).max():.4f};  "
                  f"prefix max diff {np.abs(prefix - tensors[f'{key}.prefix'][0]).max():.4f} "
                  f"(of {np.abs(tensors[f'{key}.prefix']).max():.1f})")
        return prefix

    def ask(rows: list[tuple[np.ndarray | None, list[int], list[int], dict]]) -> list[np.ndarray]:
        """Rows (prefix, ids, markers, question) laid end to end in one pass: each row's scores."""
        sizes = [(0 if prefix is None else len(prefix), len(ids)) for prefix, ids, _, _ in rows]
        seq_len = next(s for s in seq_lens if s >= sum(p + t for p, t in sizes))
        plan = hostio.layout(sizes, seq_len)
        hidden = np.zeros((1, cfg.hidden_size, 1, seq_len), np.float32)
        type_add = np.zeros((1, cfg.hidden_size, 1, seq_len), np.float32)
        for start, (p, t), (prefix, ids, _, question) in zip(plan["starts"], sizes, rows):
            if p:
                hidden[0, :, 0, start:start + p] = prefix.T
            hidden[0, :, 0, start + p:start + p + t] = embeddings[ids].T
            type_add[0, :, 0, start + p:start + p + t] = type_embeddings[hostio.QTYPES[question["type"]]][:, None]
        feeds = {"mask": plan["mask"], "head_mask": plan["head_mask"], "type_add": type_add,
                 "conv_left": hostio.wide(plan["conv_left"], cfg.hidden_size), "conv_right": hostio.wide(plan["conv_right"], cfg.hidden_size)}
        for run, names in chains[seq_len]:
            hidden = run.run(None, {"hidden_in": hidden, **{n: feeds[n] for n in names if n != "hidden_in"}})[0]
        return [hidden[0, 0, 0, [start + p + m for m in markers]] for start, (p, _), (_, _, markers, _) in zip(plan["starts"], sizes, rows)]

    worst = {"alone": 0.0, "packed": 0.0}
    flipped = 0
    for case in reference["cases"]:
        picture = case["image"] is not None
        print(f"  {case['name']}")
        prefix = see(args.reference / "images" / f"{case['image']}.png", f"image.{case['name']}") if picture else None
        if picture and len(prefix) != case["prefix"]:
            print(f"      the checkpoint's code reads this picture as {case['crops']} crops, {case['prefix']} positions; here it is one, {len(prefix)}")
        limit = min(raw["image_text_length"] if picture else raw["max_length"], seq_lens[-1] - (len(prefix) if picture else 0))
        rows = []
        for name, question in case["questions"].items():
            ids, markers = hostio.encode_row(encode, tokenizer.token_to_id, raw["bos_token_id"], case["state"], question, limit, picture)
            theirs = case["rows"][name]
            if ids != theirs["ids"] or markers != theirs["markers"]:
                print(f"      {name}: TOKENS DIFFER ({len(ids)} against {len(theirs['ids'])})")
            rows.append((prefix, ids, markers, question))
        passes = {"alone": [ask([row])[0] for row in rows]}
        if not picture and len(rows) > 1 and sum(len(r[1]) for r in rows) <= seq_lens[-1]:
            passes["packed"] = ask(rows)
        for way, scores in passes.items():
            for (name, question), z in zip(case["questions"].items(), scores):
                mine = hostio.answer(question, z, reference["temperatures"], picture)
                theirs = case["answers"][name]
                diff = float(np.abs(np.array(probabilities(mine)) - np.array(probabilities(theirs))).max())
                worst[way] = max(worst[way], diff)
                same = np.argmax(probabilities(mine)) == np.argmax(probabilities(theirs))
                flipped += not same
                print(f"      {way:6s} {name:10s} max probability diff {diff:.5f}{'' if same else '   DECISION DIFFERS'}")
    print(f"worst probability difference: alone {worst['alone']:.5f}, packed {worst['packed']:.5f}; {flipped} decisions differ")
    return 1 if flipped or max(worst.values()) > 0.02 else 0


if __name__ == "__main__":
    sys.exit(main())
