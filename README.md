# Running System-1 Decision Models on SiMa.ai MLSoC Modalix

[Laya](https://github.com/NandhaKishorM/laya) is a family of non-autoregressive "System 1"
decision models: a ModernBERT-style encoder with a small decision head. A model takes a state
(text or JSON) and a typed question (`choice`, `score` or `noul`) and returns a probability
for each option in one forward pass, with no text generation.

This repository compiles those models for the SiMa.ai Modalix MLA and runs them there:

- **`laya_sima/`** builds a checkpoint as one MLA graph on the LLiMa compiler framework
  (`sima_lmm`'s `BaseModel` + `OnnxBuilder`, the same base its Whisper encoder and vision
  towers use) and drives the same passes `llima-compile` does.
- **`runtime/`** is a standalone C++ runtime for the board. It uses only the LLiMa runtime
  library (`sima-lmm-dev`: MLA buffers, ELF loading, tokenizer).
- **`webapp/`** is an example web application on top of the runtime: a question page, a games
  page where a specialized model plays a dino runner in real time, and a model manager.

Compiled artifacts and checkpoints are not in the repository; the Build section regenerates
them.

## Results

Measured on a Modalix DevKit (eLxr 2.1.3), weights and activations in bfloat16. Latency is one
decision, end to end inside the runtime (embedding lookup + MLA + decode), mean of 20-200 runs.
Every graph is a single MLA stage: the compiler reports `MLA: 1, A65: 0`.

| checkpoint | encoder | 64 | 128 | 256 | 512 | 1024 tokens |
|---|---|---|---|---|---|---|
| `laya` (English) | ModernBERT-large, 421M | 18.0 ms | **19.5 ms** | 35.2 ms | 87.5 ms | not trained for it |
| `laya-typed-decisions` | ModernBERT-large, 421M | | **19.4 ms** | | 87.5 ms | not measured |
| `laya-multilingual` | mmBERT-base, 322M | | **8.1 ms** | **15.3 ms** | 33.9 ms | 116.5 ms |
| `laya-dino` (game head) | ModernBERT-large, 421M | **18.0 ms** | | | | |

So the English models answer in under 30 ms for inputs up to 128 tokens (question, options
and state together), and the multilingual model up to 256 tokens. Longer graphs exist for long
inputs, above that budget.

Agreement with the fp32 PyTorch model on 100 decisions (20 states x 5 questions,
`tools/agreement.py`), 128-token graphs:

| checkpoint | weights | latency | same decision | probability drift (mean / p95 / max) |
|---|---|---|---|---|
| `laya` | bf16 | 19.5 ms | 100 / 100 | 0.010 / 0.030 / 0.16 |
| `laya` | int8 encoder MLP, rest bf16 | 16.4 ms | 97 / 100 | 0.016 / 0.061 / 0.11 |
| `laya` | int8 encoder, bf16 head | 14.8 ms | 98 / 100 | 0.022 / 0.070 / 0.20 |
| `laya` | int8 everything | 14.0 ms | 98 / 100 | 0.022 / 0.070 / 0.20 |
| `laya-typed-decisions` | bf16 | 19.4 ms | 99 / 100 | 0.006 / 0.018 / 0.03 |
| `laya-multilingual` | bf16 | 8.1 ms | 100 / 100 | 0.004 / 0.017 / 0.04 |

Things worth knowing before choosing what to compile:

- **Latency is mostly weight traffic, not sequence length.** Halving the sequence from 128 to
  64 tokens saves 1.5 ms; halving the weight bytes (int8) saves 5.5 ms.
- **Int8 weights trade decisions for speed.** `--int8_weights mlp|encoder` stores only the
  chosen encoder matrices as int8 inside the same single ELF. It is faster, but the decision
  flips come from the encoder, so keeping the head in bf16 does not win them back. Int8 uses
  no calibration here; calibrated quantization (GPTQ-style, as LLiMa's model zoo uses) is the
  untried next step. A 256-token int8 build crashed the compiler's own simulator, and int8
  broke the fine-tuned game head (28/30 states).
- **Graphs load into memory reserved for the MLA** (16 GB on the DevKit, separate from Linux
  RAM and shared by every MLA application). All four checkpoints, ten graphs and about 7.4 GB
  in total, were loaded at once.
- **Loading takes seconds.** About 3 s for one English graph, 6 s for all three, 9-10 s for
  the multilingual model (its 34 MB tokenizer and 393 MB embedding table dominate).
- **Each question is one forward pass.** A request with three questions takes three passes.
- **Zero-shot quality is the checkpoint's.** The port reproduces PyTorch's answers, including
  its wrong ones; upstream describes the base models as "a fast base to specialise".
- Tokenizing adds 1-2 ms on the A65 for a short ticket, outside the bench numbers above.

## Layout

```
laya_sima/            compiler package (runs in the Model Compiler venv)
  config.py           checkpoint -> LayaConfig
  weights.py          checkpoint tensors under the names OnnxBuilder asks for
  model.py            LayaModel(BaseModel): the graph
  hostio.py           the CPU side of the contract, in numpy (reference for runtime/)
  cli.py              laya-compile
bin/laya-compile      wrapper that runs the CLI under the Model Compiler venv
bin/laya-deploy       copy models + runtime + web app to a board and set it up
runtime/              C++ runtime and `laya` CLI for the board
webapp/               example web app (stdlib Python server; question, games and model pages)
games/                the dino game's observation format and optimal policy
setup.sh, run.sh      board-side: build once, start the app
tools/                reference dump, ONNX check, board parity, agreement, I/O shape probe,
                      game-head training and its exhaustive board check
```

`models/`, `build/`, `.venv-ref/`, `.venv-train/` and `third_party/` are local and ignored by
git.

## How the graph is arranged

The graph uses LLiMa's token layout, NCHW with one token per W position:

| | tensor | shape | |
|---|---|---|---|
| in | `embeds` | (1, hidden, 1, S) | token embeddings (looked up on the CPU) |
| in | `global_mask` | (1, S, 1, S) | additive attention mask: padding |
| in | `local_mask` | (1, S, 1, S) | padding + ModernBERT's 64-token sliding window |
| in | `qtype` | (1, 16, 1, S) | question type, one-hot, repeated per position |
| out | `scores` | (1, 1, 1, S) | scorer output at every position |
| out | `act_pre` | (1, 256, 1, 1) | act head, first layer, pooled-state part |

Three things differ from the PyTorch model so that nothing runs on the CPU that does not have
to, and nothing needs an operator the MLA lacks:

- **RoPE** is applied after the heads are split, where `rotate_half` is a swap of two channel
  halves; its sign is folded into the sin table.
- **The option gather is inverted.** PyTorch gathers the hidden state at the `[MASK]` positions
  and scores those. Here the scorer runs at every position and the CPU reads the marker
  positions out of the result.
- **The act head is split by input.** Its first layer takes the pooled state plus four
  features computed from the option probabilities. The pooled-state columns run on the MLA;
  the four features and the 256->2 tail are a few hundred multiplies on the CPU.

The question-type input is padded from 3 to 16 channels because the MLA tessellation asserts
on a 3-channel input tensor (`tools/probe_io.py` reproduces this in seconds).

## Build

Prerequisites on the host: the SiMa Model Compiler with `sima-lmm` (provides `llima-compile`
and the `sima_lmm` package), usable from the NEAT SDK container. Not vendored here.

```bash
# 1. Checkpoint (the English one is at the top of the Hugging Face repo; the others are in
#    its typed-decisions/ and multilingual/ folders, with the same five files)
hf download convaiinnovations/laya model.safetensors rl_agent_config.json \
    encoder/config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json \
    --local-dir models/laya

# 2. Compile, inside the SDK container (about 15-20 minutes per sequence length)
bin/laya-compile models/laya -o build/laya --seq_lens 128,256
#    --precision A_BF16_W_INT8 for int8 weights, --int8_weights mlp|encoder for int8 on part
#    of the encoder only; --onnx / --quantize / --compile / --devkit run a single pass, as
#    with llima-compile

# 3. Deploy to the board and build the runtime there (-> /media/nvme/laya)
bin/laya-deploy build/laya --game build/laya-dino \
    --extra multilingual=build/laya-multilingual --board sima@<board-ip>
```

Sequence lengths up to about 1800 tokens compile as is. Beyond that the LLiMa builder stops
packing all attention heads into one tensor, which the RoPE step here assumes.

`build/laya/sima_files/devkit/` is what gets deployed: the ELFs, `tokenizer.json`, the token
embedding table as bfloat16, the act-head tail and `laya_config.json`.

## Run on the board

```bash
cd /media/nvme/laya
./setup.sh            # once: checks prerequisites, builds the runtime (add --check to test the MLA)
./run.sh              # http://<board-ip>:8095
./run.sh --stop
```

The app has three pages:

- **Questions** (`/`): type a state, add `choice` / `score` / yes-no questions, and see the
  decision and its latency. It opens blank; examples are one click away, and scenarios can be
  saved in the browser. A selector picks the model, and for a model with several graphs either
  the smallest one that fits or one pinned graph.
- **Games** (`/games`): Dino Arena, below. It does not start until its model is loaded.
- **Models** (`/models`): what is on the MLA, with Load and Unload per model and a choice of
  which graphs to load.

Nothing is loaded or unloaded automatically. `./run.sh` loads the general model at startup
(`--preload NAMES|all|none` changes that); every other model, the game's included, is loaded
from the Models page or from the Load button a page shows when its model is missing.
`./run.sh --help` lists the remaining options. In the API, `POST /api/predict` takes
`{"state", "questions"}` plus an optional `"model"` and `"seq_len"`; `POST /api/models/load`
and `/api/models/unload` take `{"model"}`; `GET /api/info` reports the state.

`./run.sh cli ...` runs the command-line runtime:

```bash
./run.sh cli run model --state "We were billed twice for March." \
    --question '{"type": "noul", "instructions": "Is this a billing problem?"}'
./run.sh cli bench model --iters 100          # latency of the largest graph
./run.sh cli serve model                      # one JSON request per stdin line
```

Requests and answers have the shape of upstream's `Agent.system_one`: `{"state": ...,
"questions": {id: {"type", "instructions", "criteria"}}}` in, `{"answers": {...}, "usage":
{...}}` out. `usage` carries the token count, the graph used and the latency breakdown.

If a model fails to load with `MLA_LOAD_FAILED`, the kernel could not supply a contiguous
buffer (`dmesg` shows `dma_alloc_coherent ... failed`). The usual cause is page cache
occupying the CMA pool after large ELF files were read or copied; the runtime evicts each ELF
from the page cache around loading it, and `run.sh` drops the cache at startup, for that
reason. A load that has failed leaves memory held by the MLA server, and then only
`./run.sh --reset-mla` helps: it restarts the board-wide MLA services
and so unloads every model on the MLA, other applications' included. It never runs unless
asked for.

## Dino Arena: the games harness

`/games` runs a dino runner in the browser. One lane is played by Laya, the other by you on
the keyboard, on the same course. For each decision the page sends the game state to the board
and applies the action that comes back, about 45 times a second. A slider adds delay to every
decision; in simulation the dinosaur never crashes at up to 80 ms per decision and crashes
regularly from 150 ms.

- The harness describes the nearest obstacle in words, for example
  `{"obstacle": "low bird", "timing": "now", "speed": "fast"}`. Turning a distance into
  `far` / `now` / `passing` is arithmetic done by the harness (`games/dino_policy.py`).
- The model answers one `choice` question: run, jump or duck. It has to know that a cactus
  arriving now means jump, a low bird means duck and stay down, and a high bird means do
  nothing.
- The base checkpoint cannot do this zero-shot (it gets 2-5 of 5 probe states right depending
  on phrasing; upstream describes it as "a fast base to specialise"). So the game uses its own
  checkpoint: `tools/train_dino.py` fine-tunes only the decision head on the game's 30 states,
  and that checkpoint is compiled as a second 64-token graph. The question page keeps using
  the unmodified model.

```bash
.venv-train/bin/python tools/train_dino.py            # -> models/laya-dino (minutes)
bin/laya-compile models/laya-dino -o build/laya-dino --seq_lens 64
bin/laya-deploy build/laya --game build/laya-dino
python3 tools/dino_check.py                           # every game state, on the board
node tools/sim_dino.js                                # the game itself, with a perfect player
```

A head trained in fp32 does not necessarily survive the MLA: bfloat16 moves ModernBERT's
encoder output by about 8% rms, and a first head that scored 201/201 in fp32 (on a finer,
numeric observation) scored 179/201 on the board. The training script therefore trains on
the encoder output as several arithmetics compute it, with noise on top, and
`tools/dino_check.py` tests the deployed model on every state the game can produce.

A game is an object with `reset`, `step`, `observe` and `draw` in `webapp/static/games.html`
plus a policy module like `games/dino_policy.py`; adding one means writing those and training
a head for it.

## Verify

```bash
# PyTorch golden outputs (needs torch + upstream laya)
python3 -m venv .venv-ref && .venv-ref/bin/pip install torch laya
.venv-ref/bin/python tools/reference.py

# Generated ONNX vs PyTorch, on the host (Model Compiler venv)
python tools/verify_onnx.py --seq_len 128            # max logit difference ~5e-6

# Compiled ELF on the board vs PyTorch
python3 tools/board_parity.py                        # raw logits from reference token ids
python3 tools/board_parity.py --text                 # full text path, including tokenization
.venv-ref/bin/python tools/agreement.py reference && python3 tools/agreement.py board
```

`tools/reference.py`, `tools/verify_onnx.py`, `tools/board_parity.py` and `tools/agreement.py`
take `--model` / `--model-dir` for the other checkpoints.

`laya-compile --encoder_only` and `laya hidden` are a diagnostic pair: an encoder-only graph
and a dump of its output, to see what the MLA's arithmetic does to the features a head is
trained on. The code is in place but has never been run to completion.

## Limits

- No language router: upstream picks a checkpoint per request from the text's language; here
  the model is chosen explicitly.
- Sequence lengths above 1024 tokens have not been compiled. `laya-multilingual` accepts up
  to 8192 tokens upstream; reaching that here would need the per-head attention path or
  upstream's windowing (`predict_long`), neither of which is ported.
- No batching: one question per forward pass, one request at a time.
- Not ported from upstream: `option_order`, long-document windowing (`predict_long`), hooks,
  `min_confidence` abstention and per-language temperatures.
- The shipped checkpoints' act heads saturate (act probability is 1.0 on everything tried, in
  PyTorch as well), so the act/escalate output carries little information here.

## Attribution

Laya is by Convai Innovations, Apache-2.0: https://github.com/NandhaKishorM/laya. The sequence
layout, option rendering and answer decoding in `runtime/src/sequence.cpp` are ports of
upstream's `laya/common.py` and `laya/agent.py`. The SiMa Model Compiler and the LLiMa
libraries are SiMa.ai's and are not included.
