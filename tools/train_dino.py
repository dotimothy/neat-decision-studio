#!/usr/bin/env python3
"""Specialize Laya's decision head for Dino Arena.

Laya's base checkpoint is meant to be fine-tuned per task. This trains only the decision head
(question-type embedding, the two head layers and the scorer) on every state the game can
produce, labelled by `games/dino_policy.py`; the ModernBERT encoder stays frozen, so its
output is computed once and reused for every step. It runs on a GPU if there is one and in a
few minutes on a CPU otherwise.

The MLA computes in bfloat16, and a head that merely memorizes the states in fp32 does so by
amplifying tiny differences in the encoder output; bfloat16 rounding then flips its answers
(a first attempt, on a finer-grained observation, scored 179/201 on the board while scoring
201/201 in fp32). So the head is trained on the encoder output as several arithmetics compute
it (fp32, bf16, fp16, rounded weights, with and without padding), with noise added on top,
and is checked on arithmetics it was not trained on.

    .venv-train/bin/python tools/train_dino.py --model models/laya --out models/laya-dino

The output is a checkpoint directory in the same layout as the input, ready for laya-compile.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import dino_policy
from laya.agent import Agent
from laya.common import QTYPES, build_sequence


def head_forward(model, hidden, attention_mask, marker_pos, qtype):
    """`DecisionModel.forward` from the encoder output on, without the act head."""
    h = hidden + model.type_emb(qtype)[:, None, :]
    pad = ~attention_mask.bool()
    for layer in model.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos[:, :, None].expand(-1, -1, h.size(-1))
    return model.scorer(torch.gather(h, 1, idx)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/laya"))
    ap.add_argument("--out", type=Path, default=Path("models/laya-dino"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=6.0,
                    help="stop once every state's correct logit leads by this much, under noise")
    ap.add_argument("--noise_scale", type=float, default=3.0,
                    help="training noise, as a multiple of the measured fp32-to-bf16 difference")
    ap.add_argument("--draws", type=int, default=1, help="noisy copies of each variant per step")
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = Agent(str(args.model), compile=False, device=device)
    model = agent.model.float().eval()   # eval: no dropout, the set is memorized exactly
    question = agent._to_internal(dino_policy.QUESTION)
    qtype = QTYPES[question["t"]]

    rows = dino_policy.all_states()
    sequences = [build_sequence(agent.tok, state, question, max_len=args.max_len,
                                head_max_len=agent.cfg["head_max_len"]) for state, _ in rows]
    longest = max(len(ids) for ids, _ in sequences)
    if longest >= args.max_len:
        sys.exit(f"a game state needs {longest} tokens; raise --max_len")
    pad_id = agent.tok.pad_token_id
    ids = torch.tensor([s + [pad_id] * (longest - len(s)) for s, _ in sequences], device=device)
    mask = torch.tensor([[1] * len(s) + [0] * (longest - len(s)) for s, _ in sequences], device=device)
    markers = torch.tensor([m for _, m in sequences], device=device)
    labels = torch.tensor([dino_policy.ACTIONS.index(a) for _, a in rows], device=device)
    qtypes = torch.full((len(rows),), qtype, device=device)
    print(f"{len(rows)} states, up to {longest} tokens, device {device}")

    # The encoder output for every state, computed the ways it might be computed: different
    # float formats, rounded weights, and padded out to the compiled length the way the MLA
    # graph sees it. Training on several of them (plus noise) is what makes the head indifferent
    # to which one it gets; the rest are held out to check that it generalizes to an arithmetic
    # it has not seen.
    def pad_to(length):
        extra = length - ids.size(1)
        return (torch.nn.functional.pad(ids, (0, extra), value=pad_id),
                torch.nn.functional.pad(mask, (0, extra), value=0))

    def encode(dtype, length, round_weights=None):
        saved = {k: v.clone() for k, v in model.encoder.state_dict().items()} if round_weights else None
        if round_weights:
            model.encoder.load_state_dict({k: v.to(round_weights).float() if v.is_floating_point() else v
                                           for k, v in saved.items()})
        model.encoder.to(dtype)
        padded_ids, padded_mask = pad_to(length)
        out = model.encoder(input_ids=padded_ids, attention_mask=padded_mask).last_hidden_state.float()
        model.encoder.float()
        if saved:
            model.encoder.load_state_dict(saved)
        # The head masks padding out, so only the real positions need to be kept.
        return out[:, :longest], padded_mask[:, :longest]

    with torch.no_grad():
        hidden, _ = encode(torch.float32, longest)
        before = head_forward(model, hidden, mask, markers, qtypes)
        train_sets = [encode(torch.float32, longest), encode(torch.float32, args.max_len),
                      encode(torch.bfloat16, longest), encode(torch.bfloat16, args.max_len),
                      encode(torch.float16, longest),
                      encode(torch.float32, args.max_len, round_weights=torch.bfloat16)]
        held_out = {"fp16, padded": encode(torch.float16, args.max_len),
                    "bf16 weights, bf16 activations": encode(torch.bfloat16, longest, round_weights=torch.bfloat16),
                    "fp16 weights": encode(torch.float32, longest, round_weights=torch.float16)}
    bf16_error = (train_sets[2][0] - hidden).std().item()
    noise = args.noise_scale * bf16_error
    print(f"base checkpoint: {(before.argmax(-1) == labels).float().mean().item():.1%} of states right")
    print(f"encoder output rms {hidden.std().item():.3f}; bf16 moves it by {bf16_error:.4f}; "
          f"training noise {noise:.4f}")

    def noisy(draws):
        """Every training variant of the set, `draws` times, each with its own noise."""
        h = torch.cat([h.repeat(draws, 1, 1) for h, _ in train_sets])
        m = torch.cat([m.repeat(draws, 1) for _, m in train_sets])
        copies = draws * len(train_sets)
        return (h + torch.randn_like(h) * noise, m, markers.repeat(copies, 1), qtypes.repeat(copies),
                labels.repeat(copies))

    def margins(logits, target):
        correct = logits.gather(1, target[:, None]).squeeze(1)
        others = logits.masked_fill(torch.nn.functional.one_hot(target, logits.size(1)).bool(), -1e9)
        return correct - others.max(-1).values

    trainable = [p for module in (model.type_emb, model.head, model.scorer) for p in module.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trainable:
        p.requires_grad_(True)
    # The rare actions are the ones that matter; weight classes inversely to their count.
    counts = torch.bincount(labels, minlength=len(dino_policy.ACTIONS)).float()
    loss_fn = torch.nn.CrossEntropyLoss(weight=counts.sum() / (len(counts) * counts))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    start = time.time()
    for step in range(1, args.steps + 1):
        h, m, mk, qt, target = noisy(args.draws)
        logits = head_forward(model, h, m, mk, qt)
        loss = loss_fn(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        with torch.no_grad():
            margin = margins(logits, target).min().item()
            accuracy = (logits.argmax(-1) == target).float().mean().item()
        if step % 10 == 0 or margin >= args.margin:
            print(f"step {step:4d}  loss {loss.item():.4f}  accuracy {accuracy:.1%}  worst margin {margin:+.2f}")
        if margin >= args.margin:
            break

    # Verify without noise: the real model entry point in fp32, then every variant, with the
    # head itself in bfloat16 as it will be on the MLA.
    with torch.no_grad():
        final, _ = model(ids, mask, markers, torch.ones_like(markers, dtype=torch.bool), qtypes)
        wrong = (final.argmax(-1) != labels).sum().item()
        print(f"trained in {time.time() - start:.0f}s; fp32 model {len(rows) - wrong}/{len(rows)} states right")
        for module in (model.type_emb, model.head, model.scorer):
            module.to(torch.bfloat16)
        names = ["fp32", "fp32 padded", "bf16", "bf16 padded", "fp16", "bf16 weights, padded"]
        report, bf16_margin, failed = {}, float("inf"), wrong > 0
        for name, (h, m) in list(zip(names, train_sets)) + list(held_out.items()):
            out = head_forward(model, h.to(torch.bfloat16), m, markers, qtypes).float()
            right, worst = (out.argmax(-1) == labels).sum().item(), margins(out, labels).min().item()
            seen = name in names
            print(f"  {'trained on' if seen else 'held out  '}  {name:32s} {right}/{len(rows)}  worst margin {worst:+.2f}")
            report[name] = {"right": right, "worst_margin": round(worst, 2), "held_out": not seen}
            bf16_margin = min(bf16_margin, worst)
            failed |= right != len(rows)
        for module in (model.type_emb, model.head, model.scorer):
            module.float()
    if failed:
        sys.exit("the head did not learn the policy robustly; raise --steps or --noise_scale")

    args.out.mkdir(parents=True, exist_ok=True)
    state_dict = {k: (v.half() if v.is_floating_point() and k != "temperature" else v).cpu().contiguous()
                  for k, v in model.state_dict().items()}
    save_file(state_dict, str(args.out / "model.safetensors"))
    for sub in ("encoder", "tokenizer"):
        shutil.copytree(args.model / sub, args.out / sub, dirs_exist_ok=True)
    config = json.loads((args.model / "rl_agent_config.json").read_text())
    # The base checkpoint's calibration temperatures were fitted to its own head.
    config["temperature"] = [1.0, 1.0, 1.0]
    config["temperature_by_options"] = {}
    config["max_len"] = args.max_len
    config["model_name"] = "laya-dino"
    config["training"] = {"task": "dino arena", "states": len(rows), "steps": step,
                          "worst_margin_under_noise": round(margin, 3),
                          "worst_margin_bf16_head": round(bf16_margin, 3), "noise": round(noise, 5),
                          "variants": report,
                          "trained": "decision head only",
                          "fine_tuned_from_checkpoint": True}
    (args.out / "rl_agent_config.json").write_text(json.dumps(config, indent=2))
    json.dump([{"state": s, "action": a} for s, a in rows], open(args.out / "dino_states.json", "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
