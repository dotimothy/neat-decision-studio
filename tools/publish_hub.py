#!/usr/bin/env python3
"""Publish compiled models to a Hugging Face repository, for the app's Models page to fetch.

Each model is the directory `laya-compile --devkit` writes (ELFs, embedding table, act tail,
tokenizer, laya_config.json), or for CLM the one `clm-compile --devkit` writes, uploaded
under its name. `models.json` at the top of the repository lists what is there, with sizes
and SHA-256 sums; the web app reads that list and downloads from it (webapp/server.py,
`Hub`). Laya models are listed under "models"; CLM, which an app from before it would not
know what to do with, under "models_v2", which such an app does not read.

    python3 tools/publish_hub.py --repo TDoSiMa/sima-laya                 # everything built
    python3 tools/publish_hub.py --repo TDoSiMa/sima-laya --only general  # one model
    python3 tools/publish_hub.py --repo TDoSiMa/sima-laya --list-only     # models.json and the card

Needs `huggingface_hub` and a login with write access (`hf auth login`). The list is written
from what is in the repository, so a model appears in the app once its upload has finished.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
# name on the hub (and in the app) -> build directory, and what the Models page says about it.
# Latency is one decision on a Modalix DevKit per compiled sequence length (README, Results).
MODELS = {
    "general": {
        "build": "build/laya/sima_files/devkit", "title": "Laya",
        "about": "The general question model: routing, triage, moderation and yes-or-no questions about a piece of English text.",
        "card": {"encoder": "ModernBERT-large, 421M parameters", "languages": "English",
                 "upstream": "convaiinnovations/laya", "license": "Apache-2.0"},
        "latency_ms": {"128": 19.5, "256": 35.2, "512": 87.5}, "agreement": "100 / 100"},
    "general-int8": {
        "build": "build/laya-w8/sima_files/devkit", "title": "Laya INT8",
        "about": "The general model with its weights stored as 8-bit integers: 28% faster and a third smaller, slightly less exact.",
        "card": {"encoder": "ModernBERT-large, 421M parameters", "languages": "English",
                 "upstream": "convaiinnovations/laya", "license": "Apache-2.0"},
        "latency_ms": {"128": 14.0}, "agreement": "98 / 100"},
    "typed-decisions": {
        "build": "build/laya-typed-decisions/sima_files/devkit", "title": "Laya typed-decisions",
        "about": "Fine-tuned by upstream on typed-decision workflows, with graphs for inputs of up to 1024 tokens.",
        "card": {"encoder": "ModernBERT-large, 421M parameters", "languages": "English",
                 "upstream": "convaiinnovations/laya", "license": "Apache-2.0"},
        "latency_ms": {"128": 19.4, "512": 87.5, "1024": 282.0}, "agreement": "99 / 100"},
    "multilingual": {
        "build": "build/laya-multilingual/sima_files/devkit", "title": "Laya multilingual",
        "about": "The same questions in more than 100 languages, on a smaller encoder that answers about twice as fast.",
        "card": {"encoder": "mmBERT-base, 322M parameters", "languages": "100+ languages",
                 "upstream": "convaiinnovations/laya", "license": "Apache-2.0"},
        "latency_ms": {"128": 8.1, "256": 15.3, "512": 33.9, "1024": 116.5}, "agreement": "100 / 100"},
    "chess": {
        "build": "build/laya-chess/sima_files/devkit", "title": "Laya-chess",
        "about": "LayaChess (datafreak/laya-chess): Laya fine-tuned on two million Stockfish-rated moves, to rate a chess move's win chance.",
        "card": {"encoder": "ModernBERT-large, 421M parameters", "languages": "English (chess positions)",
                 "upstream": "datafreak/laya-chess", "license": "Apache-2.0"},
        "latency_ms": {"256": 34.8}, "agreement": "same best move in 29 / 30 positions"},
    # CLM is published with its 128-token chain alone: the 32-token one is the whole encoder
    # again for a pass that is not half as long (README).
    "clm": {
        "build": "build/clm/sima_files/devkit", "chains": ["128"], "title": "CLM v0.1 8B",
        "about": "A Contrastive Language Model: a frozen Qwen3-8B reads the state and each option, and two small heads compare them. "
                 "The same three kinds of question as Laya on an encoder twenty times the size; a new question takes about 0.3 s.",
        "card": {"encoder": "Qwen3-8B, 8.2B parameters, as a chain of 18 graphs", "languages": "English",
                 "upstream": "Contrastive-LM/CLM-v0.1-8B", "license": "Apache-2.0"},
        "latency_ms": {"128": 254.0}, "agreement": "9 of 13 reference decisions (int8 weights; an approximation)"},
    "dino": {
        "build": "build/laya-dino/sima_files/devkit", "title": "Laya-dino",
        "about": "The English model with its decision head fine-tuned to play the Dino Arena game: run, jump or duck.",
        "card": {"encoder": "ModernBERT-large, 421M parameters", "languages": "English (game states)",
                 "upstream": "convaiinnovations/laya", "license": "Apache-2.0"},
        "latency_ms": {"64": 18.0}, "agreement": "30 / 30 game states"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


CONFIGS = ("laya_config.json", "clm_config.json")


def staged(name: str, spec: dict) -> Path:
    """The directory to publish: the build's own, or for a model published with only some of
    its chains (`chains`), a copy of it with those chains' files (hard links) and a
    configuration that lists them alone."""
    directory = ROOT / spec["build"]
    if "chains" not in spec or not (directory / "clm_config.json").is_file():
        return directory
    config = json.loads((directory / "clm_config.json").read_text())
    config["elfs"] = {length: config["elfs"][length] for length in spec["chains"]}
    stage = ROOT / "build" / f"publish-{name}"
    if stage.is_dir():
        for old in stage.iterdir():
            old.unlink()
    stage.mkdir(parents=True, exist_ok=True)
    keep = [file for files in config["elfs"].values() for file in files] + [
        config["token_embeddings"], config["tokenizer"], config["heads"], str(Path(config["heads"]).with_suffix(".json"))]
    for file in keep:
        try:
            (stage / file).hardlink_to(directory / file)
        except OSError:
            import shutil
            shutil.copy(directory / file, stage / file)
    (stage / "clm_config.json").write_text(json.dumps(config, indent=4))
    return stage


def entry(name: str, spec: dict) -> dict:
    """The list entry of one model: its files, and its graphs by sequence length."""
    directory = staged(name, spec)
    config = json.loads(next(directory / file for file in CONFIGS if (directory / file).is_file()).read_text())
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(directory.iterdir()) if path.is_file()]
    return {"title": spec["title"], "about": spec["about"], "card": spec["card"], "path": name,
            "kind": config.get("kind", "laya"), "checkpoint": config["model"], "precision": config["precision"],
            "graphs": config["elfs"], "latency_ms": spec["latency_ms"], "agreement": spec["agreement"],
            "files": files}


def card(repo: str, models: dict) -> str:
    def size(entry):
        return f"{sum(file['bytes'] for file in entry['files']) / 1e9:.2f} GB"

    rows = "\n".join(
        f"| `{name}` | {entry['title']} | {entry['precision']} | "
        + ", ".join(f"{s}: {ms:g} ms" for s, ms in entry["latency_ms"].items())
        + f" | {entry['agreement']} | {size(entry)} |"
        for name, entry in models.items())
    return f"""---
license: apache-2.0
base_model:
- convaiinnovations/laya
- Contrastive-LM/CLM-v0.1-8B
- Qwen/Qwen3-8B
library_name: sima-lmm
tags:
- sima.ai
- modalix
- mla
- edge
- decision-model
- modernbert
---

# Decision models (Laya and CLM), compiled for the SiMa.ai MLSoC Modalix

[Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations is a "System-1" decision
model: an encoder (ModernBERT-large, or mmBERT-base for the multilingual one) with a decision
head that answers a typed question about a piece of text in one forward pass. It picks one of
several options, scores on a scale, or says yes or no.

These are those checkpoints compiled for the Modalix MLA. Every graph is a single MLA stage,
with no layers on the CPU, and a decision takes about 19 ms at 128 tokens (8 ms for the
multilingual model).

| folder | model | precision | latency per decision, by tokens | same decision as PyTorch | size |
|---|---|---|---|---|---|
{rows}

`clm` is a second kind of decision model, [CLM v0.1 8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
by Contrastive-LM: a frozen [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) embeds the state
and each option, and two small projection heads compare them. Its encoder is compiled as a
chain of 18 graphs with int8 weights (7.25 GB on the MLA), and its latency above is one pass
through them, which holds a question's state and all its options: about 0.3 s for a new
question, nothing for one seen before. With int8 weights it is an approximation of the
PyTorch model (embedding cosine about 0.99; 9 of 13 reference decisions the same), and a
one-token option is embedded poorly.

Latency is one decision on a Modalix DevKit, measured inside the runtime. "Same decision" is
against the fp32 PyTorch model on 100 decisions with the 128-token graph. Precision `BF16` is
weights and activations in bfloat16; `A_BF16_W_INT8` keeps bfloat16 activations and stores
the weights as int8.

## What is in a folder

| file | what it is |
|---|---|
| `laya_s<N>_stage1_mla.elf` | the compiled graph for sequences of up to N tokens |
| `token_embeddings.bf16` | the embedding table, looked up on the CPU |
| `act_tail.f32` | the last layer of the act head, run on the CPU |
| `tokenizer.json` | the checkpoint's tokenizer |
| `laya_config.json` | token ids, calibration temperatures, and which graphs there are |

and in `clm`:

| file | what it is |
|---|---|
| `qwen_s128_l<NN>n2_stage1_mla.elf` | two layers of the encoder, from layer NN: 18 graphs that run one after another |
| `token_embeddings.bf16` | Qwen3-8B's embedding table, looked up on the CPU |
| `heads.f32`, `heads.json` | CLM's state and action heads, run on the CPU, and their layout |
| `tokenizer.json` | Qwen3-8B's tokenizer |
| `clm_config.json` | which graphs there are, in order |

`models.json` lists every folder with file sizes and SHA-256 sums.

## Using them

The files run on a Modalix board with the runtime and web app from
[neat-decision-studio](https://github.com/dotimothy/neat-decision-studio): its Models page downloads a model from
this repository onto the board and loads it on the MLA. By hand:

```bash
hf download {repo} --include "general/*" --local-dir .
./laya run general --state "We were billed twice for March." \\
    --question '{{"type": "noul", "instructions": "Is this a billing problem?"}}'
```

They are not usable with PyTorch or ONNX Runtime; for that, use the original checkpoints at
[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya).

## License and attribution

Apache-2.0. The models are Convai Innovations' Laya checkpoints
(https://github.com/NandhaKishorM/laya, Apache-2.0), compiled without retraining. The
exception is `dino`, whose decision head was fine-tuned on top of the English checkpoint for
a browser game. Zero-shot quality is the checkpoints': the port reproduces PyTorch's answers,
including its wrong ones.

`clm` holds the projection heads of Contrastive-LM's CLM-v0.1-8B
(https://github.com/Contrastive-LM/CLM, Apache-2.0), unchanged, and Alibaba Cloud's Qwen3-8B
(https://huggingface.co/Qwen/Qwen3-8B, Apache-2.0): its embedding table and tokenizer
unchanged, and its 36 decoder layers compiled for the MLA with their weights rounded to int8
and no language-model head.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. TDoSiMa/sima-laya")
    ap.add_argument("--only", nargs="+", choices=list(MODELS), help="upload only these models")
    ap.add_argument("--list-only", action="store_true", help="upload no model, only models.json and the card")
    ap.add_argument("--private", action="store_true", help="create the repository private")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for name in ([] if args.list_only else args.only or list(MODELS)):
        if not any((ROOT / MODELS[name]["build"] / file).is_file() for file in CONFIGS):
            sys.exit(f"{name}: no compiled model in {ROOT / MODELS[name]['build']}")
        directory = staged(name, MODELS[name])
        print(f"uploading {name} from {directory}", flush=True)
        api.upload_folder(repo_id=args.repo, folder_path=str(directory), path_in_repo=name,
                          commit_message=f"Add {name}")

    # The list is the one in the repository, with the entries of the models uploaded now
    # written again from their builds. A model that was not uploaded now keeps the entry it
    # has: its build here may have been made again since, and the sums in the list have to
    # be those of the files in the repository.
    present = set(api.list_repo_files(args.repo))
    listed = {}
    if "models.json" in present:
        from huggingface_hub import hf_hub_download
        old = json.loads(Path(hf_hub_download(args.repo, "models.json", force_download=True)).read_text())
        listed = {**old.get("models", {}), **old.get("models_v2", {})}
        for known in listed.values():
            known.setdefault("kind", "laya")
    for name in ([] if args.list_only else args.only or list(MODELS)):
        directory = staged(name, MODELS[name])
        names = [path.name for path in directory.iterdir() if path.is_file()]
        if names and all(f"{name}/{file}" in present for file in names):
            listed[name] = entry(name, MODELS[name])
            print(f"listing {name}", flush=True)
    listed = {name: listed[name] for name in [*MODELS, *listed] if name in listed}     # in the order above
    # An app from before CLM reads "models" and would not know a CLM entry: those go apart.
    manifest = json.dumps({"version": 1, "models": {name: e for name, e in listed.items() if e["kind"] == "laya"},
                           "models_v2": {name: e for name, e in listed.items() if e["kind"] != "laya"}}, indent=1).encode()
    api.upload_file(repo_id=args.repo, path_or_fileobj=manifest, path_in_repo="models.json",
                    commit_message="Update the model list")
    api.upload_file(repo_id=args.repo, path_or_fileobj=card(args.repo, listed).encode(), path_in_repo="README.md",
                    commit_message="Update the model card")
    print(f"https://huggingface.co/{args.repo}: {', '.join(listed) or 'no models yet'}")


if __name__ == "__main__":
    main()
