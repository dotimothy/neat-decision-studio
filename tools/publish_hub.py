#!/usr/bin/env python3
"""Publish compiled models to a Hugging Face repository, for the app's Models page to fetch.

Each model is the directory `laya-compile --devkit` writes (ELFs, embedding table, act tail,
tokenizer, laya_config.json), or for CLM the one `clm-compile --devkit` writes, uploaded
under its name. `models.json` at the top of the repository lists what is there, with sizes
and SHA-256 sums; the web app reads that list and downloads from it (webapp/server.py,
`Hub`). Laya models are in that repository and listed under "models", which every app reads.

CLM and LiquidAI's d1 models each have a repository of their own (`--clm-repo`, `--d1-repo`):
CLM because it is another model family, of another size, and d1 because its licence is
another as well (the LFM Open License v1.0), which that repository says and carries. They
are listed under "models_v3" with their repository's name, and the app fetches them from
there; an app from before that reads only "models" and "models_v2", and does not see them.

    python3 tools/publish_hub.py --repo dotimothy/sima-laya                 # everything built
    python3 tools/publish_hub.py --repo dotimothy/sima-laya --only general  # one model
    python3 tools/publish_hub.py --repo dotimothy/sima-laya --only clm      # goes to --clm-repo
    python3 tools/publish_hub.py --repo dotimothy/sima-laya --only d1 d1-3b # these two go to --d1-repo
    python3 tools/publish_hub.py --repo dotimothy/sima-laya --list-only     # models.json and the cards

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
LFM = "LFM Open License v1.0: commercial use is limited to organisations under 10 million dollars of annual revenue"
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
        "build": "build/clm/sima_files/devkit", "chains": ["128"], "repo": "clm", "title": "CLM v0.1 8B",
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
    # d1: latency is one pass of the trunk, which holds a picture and several questions about it.
    "d1": {
        "build": "build/d1-omni/sima_files/devkit", "repo": "d1", "title": "d1 Omni 600M",
        "about": "LiquidAI's small d1: the same three kinds of question, about a text or about a picture. "
                 "A question about a new picture takes about 0.1 s.",
        "card": {"encoder": "LFM2.5-Encoder-350M and SigLIP2, 475M parameters", "languages": "15 languages, by upstream's card",
                 "upstream": "LiquidAI/d1-omni-600M", "license": LFM},
        "latency_ms": {"128": 10.0, "512": 33.0}, "agreement": "26 of 27 reference decisions (int8 weights)",
        "vision_ms": 48},
    "d1-3b": {
        "build": "build/d1-3b/sima_files/devkit", "repo": "d1", "title": "d1 3B",
        "about": "LiquidAI's large d1, LFM2.5-VL-3B asked as a decision model: questions about a text or a picture, "
                 "read off the next token's logits. Three questions about a new picture take about 0.5 s.",
        "card": {"encoder": "LFM2.5-VL-3B with a SigLIP2 vision tower, 3.1B parameters, as a chain of 8 graphs",
                 "languages": "16 languages, by upstream's card", "upstream": "LiquidAI/d1-3B", "license": LFM},
        "latency_ms": {"128": 92.0, "384": 228.0, "512": 310.0}, "agreement": "27 of 27 reference decisions (int8 weights)",
        "vision_ms": 189},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


CONFIGS = ("laya_config.json", "clm_config.json", "d1_config.json")


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


def entry(name: str, spec: dict, repo: str | None = None) -> dict:
    """The list entry of one model: its files, and its graphs by sequence length. `repo` is
    the repository its files are in, when that is not the one the list is in."""
    directory = staged(name, spec)
    config = json.loads(next(directory / file for file in CONFIGS if (directory / file).is_file()).read_text())
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(directory.iterdir()) if path.is_file()]
    return {"title": spec["title"], "about": spec["about"], "card": spec["card"], "path": name,
            "kind": config.get("kind", "laya"), "checkpoint": config["model"], "precision": config["precision"],
            "graphs": config["elfs"], "latency_ms": spec["latency_ms"], "agreement": spec["agreement"],
            **({"repo": repo} if repo else {}), **({"vision_ms": spec["vision_ms"]} if "vision_ms" in spec else {}),
            "files": files}


def size(entry):
    return f"{sum(file['bytes'] for file in entry['files']) / 1e9:.2f} GB"


def card(repo: str, models: dict) -> str:
    elsewhere = {entry["kind"]: entry["repo"] for entry in models.values() if "repo" in entry}
    models = {name: entry for name, entry in models.items() if "repo" not in entry}
    others = ""
    if "clm" in elsewhere:
        others += f"""
[CLM v0.1 8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B), a second kind of decision
model on Qwen3-8B, is compiled the same way and published at
[{elsewhere['clm']}](https://huggingface.co/{elsewhere['clm']}).
"""
    if "d1" in elsewhere:
        others += f"""
LiquidAI's d1 models, which also read pictures, are published at
[{elsewhere['d1']}](https://huggingface.co/{elsewhere['d1']}), apart from these because their
licence is another (the LFM Open License v1.0).
"""
    rows = "\n".join(
        f"| `{name}` | {entry['title']} | {entry['precision']} | "
        + ", ".join(f"{s}: {ms:g} ms" for s, ms in entry["latency_ms"].items())
        + f" | {entry['agreement']} | {size(entry)} |"
        for name, entry in models.items())
    return f"""---
license: apache-2.0
base_model:
- convaiinnovations/laya
library_name: sima-lmm
tags:
- sima.ai
- modalix
- mla
- edge
- decision-model
- modernbert
---

# Laya decision models, compiled for the SiMa.ai MLSoC Modalix

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

Latency is one decision on a Modalix DevKit, measured inside the runtime. "Same decision" is
against the fp32 PyTorch model on 100 decisions with the 128-token graph. Precision `BF16` is
weights and activations in bfloat16; `A_BF16_W_INT8` keeps bfloat16 activations and stores
the weights as int8.
{others}
## What is in a folder

| file | what it is |
|---|---|
| `laya_s<N>_stage1_mla.elf` | the compiled graph for sequences of up to N tokens |
| `token_embeddings.bf16` | the embedding table, looked up on the CPU |
| `act_tail.f32` | the last layer of the act head, run on the CPU |
| `tokenizer.json` | the checkpoint's tokenizer |
| `laya_config.json` | token ids, calibration temperatures, and which graphs there are |

`models.json` lists every folder with file sizes and SHA-256 sums, here and in the
repositories named above: the web app reads it to know what it can download.

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
"""


def clm_card(repo: str, hub: str, models: dict) -> str:
    entry = models["clm"]
    latency = " and ".join(f"{ms:g} ms for {s} tokens" for s, ms in entry["latency_ms"].items())
    return f"""---
license: apache-2.0
base_model:
- Contrastive-LM/CLM-v0.1-8B
- Qwen/Qwen3-8B
library_name: sima-lmm
tags:
- sima.ai
- modalix
- mla
- edge
- decision-model
- qwen3
---

# CLM v0.1 8B, compiled for the SiMa.ai MLSoC Modalix

[CLM v0.1 8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) by Contrastive-LM is a
"System-1" decision model, a Contrastive Language Model: a frozen
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) embeds the state and each option, and two
small projection heads compare them, to pick one of several options, score on a scale, or say
yes or no.

This is that checkpoint compiled for the Modalix MLA. The encoder is a chain of 18 graphs of
two layers each, with int8 weights and bfloat16 activations (`{entry['precision']}`, 7.25 GB on
the MLA), for rows of up to 128 tokens. One pass through them ({latency} on a Modalix
DevKit, inside the runtime) holds a question's state and all its options: about 0.3 s for a
new question, nothing for one seen before. With int8 weights it is an approximation of the
PyTorch model (embedding cosine about 0.99; 9 of 13 reference decisions the same), and a
one-token option is embedded poorly. It is {size(entry)} to download.

## What is in the folder

| file | what it is |
|---|---|
| `qwen_s128_l<NN>n2_stage1_mla.elf` | two layers of the encoder, from layer NN: 18 graphs that run one after another |
| `token_embeddings.bf16` | Qwen3-8B's embedding table, looked up on the CPU |
| `heads.f32`, `heads.json` | CLM's state and action heads, run on the CPU, and their layout |
| `tokenizer.json` | Qwen3-8B's tokenizer |
| `clm_config.json` | which graphs there are, in order |

## Using it

The files run on a Modalix board with the runtime and web app from
[neat-decision-studio](https://github.com/dotimothy/neat-decision-studio), whose Models page
downloads the model from here onto the board and loads it on the MLA; the list it reads is in
[{hub}](https://huggingface.co/{hub}), with the Laya models. By hand:

```bash
hf download {repo} --include "clm/*" --local-dir .
./laya run clm --state "We were billed twice for March." \\
    --question '{{"type": "noul", "instructions": "Is this a billing problem?"}}'
```

It is not usable with PyTorch or ONNX Runtime; for that, use the original checkpoints,
[Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) and
[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

## License and attribution

Apache-2.0. `clm` holds the projection heads of Contrastive-LM's CLM-v0.1-8B
(https://github.com/Contrastive-LM/CLM, Apache-2.0), unchanged, and Alibaba Cloud's Qwen3-8B
(https://huggingface.co/Qwen/Qwen3-8B, Apache-2.0): its embedding table and tokenizer
unchanged, and its 36 decoder layers compiled for the MLA with their weights rounded to int8
and no language-model head.
"""


def d1_card(repo: str, hub: str, models: dict) -> str:
    rows = "\n".join(
        f"| `{name}` | [{entry['checkpoint']}](https://huggingface.co/{entry['card']['upstream']}) | {entry['precision']} | "
        + ", ".join(f"{s}: {ms:g} ms" for s, ms in entry["latency_ms"].items())
        + f" | {entry['vision_ms']:g} ms | {entry['agreement']} | {size(entry)} |"
        for name, entry in models.items())
    upstream = "\n".join(f"- {entry['card']['upstream']}" for entry in models.values())
    return f"""---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model:
{upstream}
library_name: sima-lmm
pipeline_tag: image-text-to-text
tags:
- sima.ai
- modalix
- mla
- edge
- decision-model
- liquid
- lfm2.5
---

# LiquidAI's d1 decision models, compiled for the SiMa.ai MLSoC Modalix

[Open d1](https://huggingface.co/blog/LiquidAI/open-d1) by Liquid AI is two "System-1"
decision models: each answers a typed question (one of several options, a score on a scale,
yes or no) about a text or about a picture in one forward pass, generating nothing. These are
those two checkpoints compiled for the Modalix MLA. They are not Liquid AI's files and Liquid
AI has not endorsed them.

> **Licence: the [LFM Open License v1.0](LICENSE), not Apache-2.0.** It is Liquid AI's, it
> applies to these files as derivative works of the checkpoints, and a copy is in every
> folder. In short: free for research and non-commercial use, and for commercial use by an
> organisation with less than 10 million US dollars of annual revenue; commercial use above
> that needs a licence from Liquid AI. Read the licence itself before relying on this summary.

| folder | checkpoint | precision | one pass of the trunk, by tokens | a picture | same decision as PyTorch | size |
|---|---|---|---|---|---|---|
{rows}

Measured on a Modalix DevKit, inside the runtime. A pass holds a picture and several
questions about it: with `d1-3b` three questions about a new picture take about 0.5 s in all
(the picture once, then one pass of the 384-token chain), and with `d1`, whose trunk reads
the picture again with each question, about 0.2 s. "Same
decision" is against the fp32 PyTorch checkpoint on 27 reference questions, 14 of them about
pictures.

## What was changed

The weights are Liquid AI's, **modified**: rounded to int8 (activations are bfloat16,
`A_BF16_W_INT8`) and compiled into graphs for the MLA with SiMa.ai's Model Compiler, by
this repository's author. Nothing was retrained. So the answers are an approximation of the checkpoints':
on the reference questions the largest difference in a probability was 0.15 for `d1` where
the decision was the same, and 0.07 for `d1-3b`.

Left out of the port: a large picture is read as one picture of at most 512 x 512 pixels
(256 positions) where Liquid AI's code cuts it into tiles; d1-omni's audio encoder is not
compiled; a state longer than the longest graph is cut at its end.

## What is in a folder

| file | what it is |
|---|---|
| `d1_s<N>_l<LL>n<K>_stage1_mla.elf` | K layers of the trunk from layer LL, for rows of up to N tokens: one graph a length in `d1`, a chain of eight in `d1-3b` |
| `d1_vision_n1024_stage1_mla.elf`, `d1_projector_m256_stage1_mla.elf` | the vision tower, for a picture of up to 1024 patches, and its projector |
| `token_embeddings.bf16` | the embedding table, looked up on the CPU (and read against for d1-3B's logits) |
| `position_embedding.f32`, `type_embeddings.f32` | the tower's position table, and d1-omni's question types |
| `tokenizer.json` | the checkpoint's tokenizer, unchanged |
| `d1_config.json` | token ids, temperatures, and which graphs there are, in order |
| `LICENSE` | the LFM Open License v1.0, unchanged |

## Using them

The files run on a Modalix board with the runtime and web app from
[neat-decision-studio](https://github.com/dotimothy/neat-decision-studio), whose Models page
downloads a model from here onto the board and loads it on the MLA; the list it reads is in
[{hub}](https://huggingface.co/{hub}), with the other decision models. By hand:

```bash
hf download {repo} --include "d1/*" --local-dir .
./laya run d1 --state "We were billed twice for March." \\
    --question '{{"type": "noul", "instructions": "Is this a billing problem?"}}'
```

They are not usable with PyTorch or ONNX Runtime; for that, use Liquid AI's checkpoints,
[LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) and
[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B).

## Attribution

The models are Liquid AI's (Liquid AI, Inc.), released under the LFM Open License v1.0. The
compilation, the runtime and this page are not Liquid AI's.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. dotimothy/sima-laya")
    ap.add_argument("--clm-repo", default="dotimothy/sima-clm", help="where CLM goes")
    ap.add_argument("--d1-repo", default="dotimothy/sima-d1", help="where LiquidAI's d1 models go: their licence is another")
    ap.add_argument("--only", nargs="+", choices=list(MODELS), help="upload only these models")
    ap.add_argument("--list-only", action="store_true", help="upload no model, only models.json and the card")
    ap.add_argument("--private", action="store_true", help="create the repository private")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    homes = {"clm": args.clm_repo, "d1": args.d1_repo}
    home = lambda name: homes.get(MODELS[name].get("repo"), args.repo)     # the repository a model's files go to
    for name in ([] if args.list_only else args.only or list(MODELS)):
        if not any((ROOT / MODELS[name]["build"] / file).is_file() for file in CONFIGS):
            sys.exit(f"{name}: no compiled model in {ROOT / MODELS[name]['build']}")
        directory = staged(name, MODELS[name])
        print(f"uploading {name} from {directory} to {home(name)}", flush=True)
        api.create_repo(home(name), repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=home(name), folder_path=str(directory), path_in_repo=name,
                          commit_message=f"Add {name}")

    # The list is the one in the repository, with the entries of the models uploaded now
    # written again from their builds. A model that was not uploaded now keeps the entry it
    # has: its build here may have been made again since, and the sums in the list have to
    # be those of the files in the repository.
    present = {args.repo: set(api.list_repo_files(args.repo))}
    listed = {}
    if "models.json" in present[args.repo]:
        from huggingface_hub import hf_hub_download
        old = json.loads(Path(hf_hub_download(args.repo, "models.json", force_download=True)).read_text())
        listed = {**old.get("models", {}), **old.get("models_v2", {}), **old.get("models_v3", {})}
        for name, known in listed.items():
            known.setdefault("kind", "laya")
            if "repo" in known and name in MODELS:          # where it is now, should a repository have moved
                known["repo"] = home(name)
    for name in ([] if args.list_only else args.only or list(MODELS)):
        directory = staged(name, MODELS[name])
        names = [path.name for path in directory.iterdir() if path.is_file()]
        if home(name) not in present:
            present[home(name)] = set(api.list_repo_files(home(name)))
        if names and all(f"{name}/{file}" in present[home(name)] for file in names):
            listed[name] = entry(name, MODELS[name], home(name) if home(name) != args.repo else None)
            print(f"listing {name}", flush=True)
    listed = {name: listed[name] for name in [*MODELS, *listed] if name in listed}     # in the order above
    # An app from before CLM reads "models" alone and would not know a CLM entry; one from
    # before d1 reads "models_v2" too, and would refuse the whole list for an entry it does
    # not know or fetch a model from this repository that is in another. So "models_v3" holds
    # the models such apps cannot fetch: those of a kind they do not know, and those elsewhere.
    def key(e):
        return "models_v3" if "repo" in e or e["kind"] not in ("laya", "clm") else "models" if e["kind"] == "laya" else "models_v2"
    manifest = json.dumps({"version": 1, **{k: {name: e for name, e in listed.items() if key(e) == k}
                                            for k in ("models", "models_v2", "models_v3")}}, indent=1).encode()
    api.upload_file(repo_id=args.repo, path_or_fileobj=manifest, path_in_repo="models.json",
                    commit_message="Update the model list")
    api.upload_file(repo_id=args.repo, path_or_fileobj=card(args.repo, listed).encode(), path_in_repo="README.md",
                    commit_message="Update the model card")
    for other in sorted({e["repo"] for e in listed.values() if "repo" in e}):
        there = {name: e for name, e in listed.items() if e.get("repo") == other}
        if any(e["kind"] == "d1" for e in there.values()):       # LiquidAI's licence goes with its models
            licence = ROOT / MODELS[next(iter(there))]["build"] / "LICENSE"
            api.upload_file(repo_id=other, path_or_fileobj=str(licence), path_in_repo="LICENSE", commit_message="Add the licence")
        page = d1_card if any(e["kind"] == "d1" for e in there.values()) else clm_card
        api.upload_file(repo_id=other, path_or_fileobj=page(other, args.repo, there).encode(), path_in_repo="README.md",
                        commit_message="Update the model card")
        print(f"https://huggingface.co/{other}: {', '.join(there)}")
    print(f"https://huggingface.co/{args.repo}: {', '.join(listed) or 'no models yet'}")


if __name__ == "__main__":
    main()
