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
| `laya-typed-decisions` | ModernBERT-large, 421M | | **19.4 ms** | | 87.5 ms | 282 ms |
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
- **One model is loaded at a time in the app**, by choice rather than necessity (see the
  model manager below).
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
  static/neat.css      the look shared by every page; brand.js builds the header and footer
  static/brand/, fonts/ marks and fonts taken from SiMa.ai's NEAT GenAI Studio example
games/                per game: what the model is asked and why, and the reference policy
setup.sh, run.sh      board-side: build once, start the app
tools/                reference dump, ONNX check, board parity, agreement, I/O shape probe,
                      game-head training, and per game a board check and a simulation
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

The app presents itself as "Laya Decision Studio, running on SiMa.ai Palette Neat" and follows
the look of SiMa.ai's NEAT GenAI Studio example (`webapp/static/neat.css`, light and dark with
the browser's setting). `webapp/static/brand.js` builds the header, the introduction on the
landing page and the "Powered by" footer for every page, so that wording is in one place.

The app has four areas:

- **Debate** (`/`, the default): type a yes-or-no question and the model answers it by itself,
  with exactly two options, on every change to the text (a decision every 25-40 ms). A pie
  chart of the yes and no probabilities follows the typing, with the time on the MLA above it
  and a trace of how the answer moved. The text is put as `Question: ...` and asked "Is the
  answer yes or no?"; of eight wordings that one did best, 21 of 24 simple factual questions
  right on the English model. It has nothing to look things up in, so this is what the
  encoder absorbed in pre-training.
- **Questions** (`/questions`): type a state, add `choice` / `score` / yes-no questions, and
  see the decision and its latency. It opens blank; examples are one click away, and scenarios
  can be saved in the browser.
- **Games** (`/games`): Dino Arena, Blackjack, Snake and Sudoku, below.
- **Models** (`/models`): the model manager. It shows what is on the MLA and has Load and
  Unload per model, with a choice of which graphs to load, and a progress bar while a model
  loads or unloads. The bar's stages are real (the runtime logs when it starts and finishes
  each graph); inside a stage it is the time spent against a measured estimate.

One model is on the MLA at a time, and choosing and loading it happens only in the model
manager: loading a model there unloads the one that was loaded. Every other page just names
the loaded model and uses it, whichever it is, so any deployed model can be tried on any page
by loading it. With none loaded a page says so and waits, and starts by itself once one is
there. Next to the model's name each page has a **token budget**: how many tokens a question,
its options and the state may take together, up to the largest loaded graph. A longer state
is cut and the page says by how much. A game's requests are 40 to 110 tokens, so a budget
below that cuts what the model is shown and the play shows it: blackjack at a budget of 70
took the intended action in 307 of 432 decisions, against 390 of 390 uncut.

`./run.sh` loads the general model at startup (`--preload NAME|none` changes that) and logs
every load and unload with the address it came from. `./run.sh --help` lists the remaining
options. In the API, `POST /api/predict` takes `{"state", "questions"}` plus an optional
`"model"`, `"seq_len"` (pin a graph), `"max_len"` (the token budget) and `"head_max_len"`
(the part of it the question and options may take), and `usage` reports the tokens used, the
budget, the graph and the tokens cut; `POST /api/models/load` and `/api/models/unload` take
`{"model"}`; `GET /api/info` reports the state.

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

## Blackjack, Snake and Sudoku: games on a general model

These three use a Laya checkpoint as it ships, with no fine-tuning. Any deployed model can be
tried by loading it in the model manager; the numbers below are for the English one. Asking a general model
to play only works if the question is put a particular way, and how that was found is the
useful part (`games/blackjack_policy.py`, `games/snake_policy.py`, `games/sudoku_policy.py`):

- **It does no arithmetic.** Given blackjack totals ("hard 16 against a 10") it answers the
  same for every hand: 42-58% agreement with basic strategy, the base rate. The harness has
  to turn numbers into facts in words.
- **It does not weigh facts against each other.** Asked "hit or stand?" about a hand
  described in words, none of 108 wordings got all eight situations right.
- **It does match a description to a rule, or pick the best-described option.** So either the
  rules of thumb are written as the question's criteria (blackjack), or each option is
  described by what it leads to (Snake). About four rules fit in one question; a fifth
  brought blackjack down to 8 of 10.
- **Extra detail hurts.** Adding the card totals next to the facts lowered blackjack's
  agreement with basic strategy from 98% to 80-91%.

### Blackjack, with a memory

`/games/blackjack`: one deck, dealer stands on 17, blackjack pays 3 to 2, doubling on a
two-card 9, 10 or 11. The model keeps nothing between requests, so the memory is the
harness's: every card shown since the shuffle, and a Hi-Lo count from it. It reaches the
model as three questions:

| question | what the harness states | options |
|---|---|---|
| bet, before the hand | `Cards left in the shoe: rich in tens and aces.` (from the count) | 1, 3 or 6 units |
| double, on 9-11 | `My edge over the dealer if I take exactly one card: small.` (from the unseen cards) | double, or play on |
| hit or stand | `Hand: weak. Drawing a card: risky. Dealer: likely to bust.` (the last two from the unseen cards) | four rules of thumb |

On the board the model gives the intended action for all 15 distinct texts, and from a fresh
deck it plays 255 of 260 hit-or-stand hands as basic strategy does; the other five are hands
the three facts cannot tell apart (hard 12 against a 2 or 3, soft 18 against a 9, 10 or ace).
Its margins are thin in places: 0.31-0.59 on the hit-or-stand rules, and 0.46 against 0.46
between betting 3 and 6 on a rich shoe.

What the memory is worth, over two million simulated hands (`tools/sim_blackjack.js`):

| player | units won per 100 hands |
|---|---|
| basic strategy, flat bet | -0.56 |
| described facts without memory, flat bet | -0.67 |
| described facts from the remembered cards, flat bet | -0.50 |
| the same, betting 1 / 3 / 6 by the count | +1.45 |

The page shows the count, the cards left by group, each question as it is asked, the net
result next to the same play at flat bets, and the chart of what the model plays. **Reset
count** forgets the cards and shuffles a fresh deck (forgetting without shuffling would make
the memory wrong); **Reset Scores** clears the tallies.

### Snake

`/games/snake`: a 20 x 14 board unless changed, one decision per move. The model is not shown the board. For
each of the three moves (left, straight, right, relative to the heading) the harness works
out where it leads and describes it with one of four phrases: `safe and toward the food`,
`safe but away from the food`, `a dead end, the snake dies`, `blocked, the snake dies`. Those
descriptions are the options of the question, rebuilt every move, and the model picks one.
What the harness computes is geometry (is the square free, does the shortest path to the food
get shorter, is the space beyond smaller than the snake).

On the board the model picks a best-described move in all 56 combinations that offer a safe
one. A player that always does so averages 60 food per game in simulation
(`tools/sim_snake.js`); in the browser the model moved the snake about 40 times a second.

The grid is the user's to change, from 6 x 6 to 60 x 40, by preset or by typing columns and
rows; the browser remembers it, and a change starts a new game and new scores. Nothing in the
question mentions the board, so the model is asked the same thing on any grid and what
changes is the game: the same always-best player averages 15 food on 6 x 6, 29 on 10 x 8, 88
on 30 x 20 and 168 on 60 x 40 (`node tools/sim_snake.js 100 60 40`).

How much of the board goes into the descriptions can be changed too ("Laya sees"): the whole
board, or only the squares within 12, 8, 5, 3, 2 or 1 of the snake's head. With less than the
whole board the harness still knows the edges and where the food is, but of the snake's body
only the part in sight, and takes the rest to be empty, so a move can be described as safe
that is not. The page veils what is out of sight. The model's part does not change (it still
picks the best-described move); the descriptions get worse, and the always-best player on
20 x 14 drops from 61 food per game to 55 seeing 12 squares, 47 seeing 8, 36 seeing 5 and
about 26-29 seeing 3 or fewer (`node tools/sim_snake.js 300 20 14 5`).

### Sudoku

`/games/sudoku`: Laya fills a 9 x 9 puzzle, two decisions a cell. The model is not shown the
grid, because it cannot read one: given the digits a cell's row, column and box hold and asked
which digit is missing, it answered with a digit from the lists in 399 of 400 tries. It finds
what is in the text, not what is absent. So the harness does the looking, as in Snake, and
the options say what it found:

- **Which cell.** Five empty cells are offered, one of the surest on the board always among
  them, each described as `safe and certain, one digit must go here`, `a small risk, two
  digits fit` or `a big risk, many digits fit`.
- **Which digit.** The nine digits, each described as `safe and certain`, `safe but a guess`
  or `repeats in the row, breaks the puzzle` (or column, or box).

A digit is certain when it is the only one left for its cell, or when its cell is the only
place left for it in a row, column or box. The level decides what a puzzle needs: easy ones
only the first kind (and they keep 35 of their digits), medium ones both, and hard ones run
out of certain cells, so the model has to take a risk. A wrong digit is counted and replaced
by the right one, so every puzzle gets finished and what is counted is the mistakes. Puzzles
are generated in the browser, each with one solution.

On the board the English model takes a certain cell whenever one is offered (300 of 300
combinations) and a best-described digit in all 301 cells tried, never one that breaks the
puzzle. What it does not do is rank the two kinds of risk: with no certain cell on offer it
takes the smaller risk in 38 of 60 combinations (typed-decisions: 58 of 60). The wording
matters as it did in Snake: with `fits` against `taken by the row` the model wrote a
rule-breaking digit whenever none was certain. A player that always takes a best-described
option solves every easy and medium puzzle without a mistake, and makes 1.2 mistakes per hard
puzzle from 2.2 risks (`tools/sim_sudoku.js`). The longest question is 120 tokens, so every
decision runs on the 128-token graph.

```bash
python3 tools/blackjack_check.py      # all 15 blackjack texts on the board, and the chart
python3 tools/snake_check.py          # all 64 combinations of move descriptions on the board
node tools/sim_blackjack.js           # what remembering the cards is worth
node tools/sim_snake.js               # what the move descriptions are worth (add: games columns rows sight)
python3 tools/sudoku_check.py         # the Sudoku cell and digit questions on the board
node tools/sim_sudoku.js              # what the descriptions are worth at each level, and the puzzles
```

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
- Sequence lengths above 1024 tokens have not been compiled; the 1024-token graphs exist for
  the two checkpoints trained for that length. `laya-multilingual` accepts up
  to 8192 tokens upstream; reaching that here would need the per-head attention path or
  upstream's windowing (`predict_long`), neither of which is ported.
- No batching: one question per forward pass, one request at a time.
- Not ported from upstream: `option_order`, long-document windowing (`predict_long`), hooks,
  `min_confidence` abstention and per-language temperatures.
- The shipped checkpoints' act heads saturate (act probability is 1.0 on everything tried, in
  PyTorch as well), so the act/escalate output carries little information here.

## License

Apache-2.0; see [LICENSE](LICENSE). That covers the code in this repository only. The SiMa
Model Compiler and the LLiMa libraries it builds on are SiMa.ai's and are licensed separately;
the Laya checkpoints are upstream's.

The NEAT mark, the SiMa.ai logos and the Inter and JetBrains Mono fonts under
`webapp/static/brand/` and `webapp/static/fonts/` are copied from SiMa.ai's NEAT GenAI Studio
example. The logos are SiMa.ai's trademarks, and the fonts are under the SIL Open Font License.

## Attribution

Laya is by Convai Innovations, Apache-2.0: https://github.com/NandhaKishorM/laya. The sequence
layout, option rendering and answer decoding in `runtime/src/sequence.cpp` are ports of
upstream's `laya/common.py` and `laya/agent.py`. The SiMa Model Compiler and the LLiMa
libraries are SiMa.ai's and are not included.
