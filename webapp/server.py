#!/usr/bin/env python3
"""Laya decision playground: a web front end for the Laya runtime on a Modalix DevKit.

Runs on the board with nothing but the Python standard library. It keeps a `laya serve`
process alive for the loaded model, so its compiled graphs stay on the MLA, and forwards each
browser request to it as one JSON line.

    /           the landing page: what this is in a line, a question to ask, and the ways in
    /showcase   the story of the demo as a deck of slides, for presenting
    /debate     a yes-or-no question answered live as it is typed
    /questions  question answering with your own questions and options
    /games      games a model plays, and people with it: Snake (the default), Chess, Tic-Tac-Toe,
                Rock Paper Scissors, Dino Arena, Blackjack, Sudoku
    /models     any page with Settings open: the model manager (every model on the board or
                on Hugging Face, with download, load, unload and delete)

Several models can be on the MLA at once, as many as fit in its memory, so that they can be
asked the same thing side by side (`POST /api/compare`). Models are loaded only on request,
in Settings; `--preload` chooses what is loaded at startup. The app also starts with no model
on the board at all: Settings then offers the ones published on Hugging Face (`--hub`),
downloads one into the app directory and lists it.

    python3 webapp/server.py --model /media/nvme/laya/model --laya /media/nvme/laya/laya
"""
import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 1 << 20
try:
    VERSION = (Path(__file__).resolve().parents[1] / "VERSION").read_text().strip()
except OSError:
    VERSION = "unknown"


class RuntimeDied(RuntimeError):
    pass


class LayaProcess:
    """One `laya serve` child. The MLA runs one graph at a time, so requests are serialized."""

    def __init__(self, laya: str, model: str, seq_lens: str | None):
        # The runtime serves either kind of model directory, Laya or CLM, the same way.
        self._command = [laya, "serve", model] + (["--seq-lens", seq_lens] if seq_lens else [])
        self._lock = threading.Lock()
        self._proc = None
        self._log: list[str] = []
        self.info: dict = {}

    def _read_json_line(self, on_line=None) -> dict:
        """Next JSON document from the child; the MLA libraries also log to stdout.

        `on_line` is called with each log line on the way, which is how loading is followed.
        """
        while True:
            line = self._proc.stdout.readline()
            if not line:
                tail = " | ".join(self._log[-4:]) or "no output"
                raise RuntimeDied(f"the Laya runtime exited ({self._proc.wait()}): {tail}")
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
            if line:
                self._log = (self._log + [line])[-20:]
                if on_line:
                    on_line(line)

    def start(self, on_line=None):
        with self._lock:
            self._proc = subprocess.Popen(
                self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            self.info = self._read_json_line(on_line)

    def request(self, payload: dict) -> dict:
        with self._lock:
            if not self.running:
                raise RuntimeDied("the Laya runtime is not running")
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
                return self._read_json_line()
            except BrokenPipeError as error:
                raise RuntimeDied("the Laya runtime closed its input") from error

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def stop(self):
        with self._lock:
            if self.running:
                # Closing stdin ends `laya serve` cleanly, which releases the model on the MLA.
                self._proc.stdin.close()
                try:
                    self._proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
            self._proc = None


class NotLoaded(RuntimeError):
    pass


class Refused(RuntimeError):
    pass


class Progress:
    """How far a load or unload has come: named stages, each with an estimate in seconds.

    The stage boundaries are real (the runtime logs when each graph starts and finishes
    loading); inside a stage the position is the time spent against the estimate, held just
    short of the end so the bar never claims a stage is done before it is.
    """

    def __init__(self, stages: list[tuple[str, float]]):
        self.stages = stages
        self.index = 0
        self._started = time.monotonic()

    def enter(self, index: int):
        if index > self.index:
            self.index, self._started = index, time.monotonic()

    def describe(self) -> dict:
        index = min(self.index, len(self.stages) - 1)
        done = sum(seconds for _, seconds in self.stages[:index])
        label, seconds = self.stages[index]
        inside = min(time.monotonic() - self._started, seconds * 0.95)
        total = sum(seconds for _, seconds in self.stages)
        return {"fraction": round((done + inside) / total, 3), "stage": label,
                "step": index + 1, "steps": len(self.stages)}


# Measured on a Modalix DevKit: seconds for each part of bringing a model up.
def seconds_to_read(tokenizer_bytes: int, embedding_bytes: int) -> float:
    return 0.3 + 0.088 * tokenizer_bytes / 1e6 + 0.0106 * embedding_bytes / 1e6


def seconds_to_load_graph(elf_bytes: int) -> float:
    return 0.1 + 1.4 * elf_bytes / 1e9


SECONDS_TO_WARM_UP = 0.25
SECONDS_TO_UNLOAD = 0.4


CONFIGS = {"laya": "laya_config.json", "clm": "clm_config.json"}


def model_kind(path: Path) -> str | None:
    """Which kind of compiled model a directory holds, if any."""
    return next((kind for kind, name in CONFIGS.items() if (Path(path) / name).is_file()), None)


class Model:
    """One compiled model directory and, while it is loaded, the runtime process holding it.

    There are two kinds. A Laya directory has one graph, in one file, for each sequence length
    it was compiled for, and any of them can be loaded. A CLM directory has one sequence
    length, whose graph is a chain of files (each a run of the encoder's layers) that are all
    loaded together.
    """

    def __init__(self, name: str, path: str, laya: str, allowed: str | None):
        self.name, self.path, self._laya = name, Path(path), laya
        self.kind = model_kind(self.path) or "laya"
        self.config = json.loads((self.path / CONFIGS[self.kind]).read_text())
        if self.kind == "clm":
            self._files = {int(self.config["seq_len"]): list(self.config["elfs"])}
            sizes = list(self._files)
        else:
            self._files = {int(s): [file] for s, file in self.config["elfs"].items()}
            sizes = sorted(self._files)
            if allowed:
                wanted = {int(s) for s in allowed.split(",")}
                sizes = [s for s in sizes if s in wanted]
        self.seq_lens = sizes                 # graphs this app may load
        self.loaded_seq_lens: list[int] = []
        # What is on the MLA right now, which during a load is not yet everything asked for:
        # the graphs that have arrived, and the one arriving with when it started and how long
        # it is expected to take.
        self.resident: list[int] = []
        self.arrived_files: list[str] = []
        # How much more the MLA held after this model was loaded than before: its share as
        # measured, where the graph files' sizes are only an estimate. None: not measured.
        self.held_bytes: int | None = None
        self.arriving: tuple[str, float, float] | None = None
        self.state = "unloaded"               # unloaded | loading | loaded | unloading
        self.progress: Progress | None = None  # while loading or unloading
        self.process: LayaProcess | None = None

    def graph_files(self, seq_len: int) -> list[str]:
        """The files of one sequence length's graph, in the order the runtime loads them."""
        return self._files[seq_len]

    def file_bytes(self, file: str) -> int:
        return (self.path / file).stat().st_size

    def graph_bytes(self, seq_len: int) -> int:
        return sum(self.file_bytes(file) for file in self.graph_files(seq_len))

    def mla_bytes(self) -> tuple[int, int]:
        """Bytes of this model on the MLA at this moment: (arrived, of the file still arriving).

        The second is an estimate: the share of the file's size that its loading time so far
        is of the time it is expected to take.
        """
        arrived = sum(self.file_bytes(file) for file in self.arrived_files)
        arriving = self.arriving
        if arriving is None:
            return arrived, 0
        file, started, seconds = arriving
        return arrived, int(self.file_bytes(file) * min(0.97, (time.monotonic() - started) / max(seconds, 0.05)))

    def other_files(self) -> list[str]:
        """What the directory holds besides its configuration and graphs."""
        keys = ("token_embeddings", "tokenizer", "heads") if self.kind == "clm" else ("token_embeddings", "act_tail", "tokenizer")
        files = [self.config[key] for key in keys]
        if self.kind == "clm":                    # the heads come with their description
            files.append(str(Path(self.config["heads"]).with_suffix(".json")))
        return files

    def fixed_bytes(self) -> int:
        """What the runtime holds besides graphs: the embedding table and the tokenizer."""
        return sum((self.path / self.config[key]).stat().st_size for key in ("token_embeddings", "tokenizer"))

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "checkpoint": self.config.get("model"), "precision": self.config.get("precision"),
            "max_len": self.config.get("max_len", self.config.get("seq_len")), "hidden_size": self.config.get("hidden_size"),
            "graphs": [{"seq_len": s, "bytes": self.graph_bytes(s), "files": len(self.graph_files(s))} for s in self.seq_lens],
            "seq_lens": self.seq_lens, "fixed_bytes": self.fixed_bytes(),
            "state": self.state, "loaded": self.state == "loaded",
            "loaded_seq_lens": self.loaded_seq_lens,
            "resident_seq_lens": list(self.resident), "mla_bytes": sum(self.mla_bytes()),
            "progress": progress.describe() if (progress := self.progress) else None,
        }

    def loading_stages(self, seq_lens: list[int], unloading: list[str]) -> list[tuple[str, float]]:
        """What loading these graphs goes through, in order, with estimates: one stage a file."""
        if self.kind == "clm":                    # the embedding table is mapped, not read in
            fixed = seconds_to_read((self.path / self.config["tokenizer"]).stat().st_size, 0) + 0.5
        else:
            fixed = seconds_to_read((self.path / self.config["tokenizer"]).stat().st_size,
                                    (self.path / self.config["token_embeddings"]).stat().st_size)
        graphs = []
        for s in seq_lens:
            files = self.graph_files(s)
            for position, file in enumerate(files):
                label = (f"Loading the {s}-token graph" if len(files) == 1
                         else f"Loading the {s}-token graph, part {position + 1} of {len(files)}")
                graphs.append((label, seconds_to_load_graph(self.file_bytes(file))))
        return ([(f"Unloading {other}", SECONDS_TO_UNLOAD) for other in unloading]
                + [("Reading the tokenizer and embeddings", fixed)] + graphs
                + [("Warming up", SECONDS_TO_WARM_UP if self.kind == "laya" else 2.0)])


def memory_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


RESERVED_MEMORY = Path("/sys/firmware/devicetree/base/reserved-memory")


def mla_memory_regions() -> list[tuple[int, int]]:
    """(address, size) of the memory reserved for MLA models (the device tree's `dms` nodes).

    Compiled graphs are loaded there, not into Linux RAM, so loading one barely moves
    MemAvailable. The region is shared by every application that uses the MLA.
    """
    def cells(name: str) -> int:
        try:
            return int.from_bytes((RESERVED_MEMORY / name).read_bytes()[:4], "big") or 2
        except OSError:
            return 2

    address_cells, size_cells = cells("#address-cells"), cells("#size-cells")
    regions = []
    for node in sorted(RESERVED_MEMORY.glob("dms@*")):
        try:
            reg = (node / "reg").read_bytes()
        except OSError:
            continue
        split = address_cells * 4
        size = int.from_bytes(reg[split:split + size_cells * 4], "big")
        if size:
            regions.append((int.from_bytes(reg[:split], "big"), size))
    return regions


def mla_memory_total() -> int:
    return sum(size for _, size in mla_memory_regions())


class MlaHeld:
    """How much of the MLA's memory is taken, by every program on the board.

    The MLA's memory is handed out by one process, the dispatcher (`mlashmcomplex`), and what
    it has taken shows in its memory map as mappings of /dev/simaai-mem inside the reserved
    region. This is the number that decides whether another graph fits. The dispatcher does
    not always give memory back when a model is unloaded, and never after a load that failed,
    so it can be high with little loaded; only resetting the accelerator brings it down.

    The dispatcher runs as root, so its map is read directly when this app is root and through
    `sudo -n` otherwise. Where neither works the answer is None and the app goes by its own
    count. The reading is kept for two seconds: pages ask often.
    """

    def __init__(self):
        self._regions = mla_memory_regions()
        self._lock = threading.Lock()
        self._at, self._value = 0.0, None
        self._sudo = shutil.which("sudo") is not None and os.environ.get("LAYA_MLA_MEMORY_SUDO", "1") != "0"

    def _maps(self) -> str | None:
        fixed = os.environ.get("LAYA_MLA_MAPS")              # a file standing in for it, off the board
        if fixed:
            try:
                return Path(fixed).read_text()
            except OSError:
                return None
        pid = None
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                try:
                    if Path(f"/proc/{entry}/comm").read_text().strip() == "mlashmcomplex":
                        pid = entry
                        break
                except OSError:
                    pass
        if pid is None:
            return None
        try:
            return Path(f"/proc/{pid}/maps").read_text()
        except OSError:
            pass
        if not self._sudo:
            return None
        try:
            done = subprocess.run(["sudo", "-n", "cat", f"/proc/{pid}/maps"], capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            # sudo wanting a password will not change: stop asking. Anything else (the
            # dispatcher restarting under a reset, say) is tried again next time.
            if "password" in done.stderr or "sudo:" in done.stderr:
                self._sudo = False
            return None
        return done.stdout

    def _pieces(self, maps: str) -> set[tuple[int, int]]:
        """The pieces of the MLA's memory in a memory map, each once however often it is mapped."""
        pieces = set()
        for line in maps.splitlines():
            parts = line.split()
            if len(parts) < 6 or not parts[5].endswith("simaai-mem"):
                continue
            try:
                low, high = (int(part, 16) for part in parts[0].split("-"))
                offset = int(parts[2], 16)
            except ValueError:
                continue
            if any(address <= offset < address + size for address, size in self._regions):
                pieces.add((offset, high - low))
        return pieces

    def read(self, fresh: bool = False) -> int | None:
        with self._lock:
            if not fresh and time.monotonic() - self._at < 2.0:
                return self._value
            self._at, self._value = time.monotonic(), None
            maps = self._maps() if self._regions else None
            if maps is not None:
                self._value = sum(size for _, size in self._pieces(maps))
            return self._value

    def clients(self, own: set[int]) -> list[dict]:
        """The other programs using the MLA: every process with a piece of its memory mapped,
        but the dispatcher and this app's own runtimes (`own`), one entry a program.

        A client maps only its input and output buffers; its models are in the dispatcher's
        memory, which does not say whose they are. So this says who is there, not how much each
        holds. Processes of other users are skipped unless this app is root.
        """
        if not self._regions or os.environ.get("LAYA_MLA_MAPS"):
            return []
        found: dict[str, dict] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or int(entry) in own or int(entry) == os.getpid():
                continue
            try:
                if Path(f"/proc/{entry}/comm").read_text().strip() == "mlashmcomplex":
                    continue
                maps = Path(f"/proc/{entry}/maps").read_text()
                if "simaai-mem" not in maps or not self._pieces(maps):
                    continue
                command = Path(f"/proc/{entry}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                continue
            name = program_name(command)
            found.setdefault(name, {"name": name, "pids": [], "command": command[:160]})["pids"].append(int(entry))
        return list(found.values())


def program_name(command: str) -> str:
    """What to call a program on the MLA, from its command line."""
    for mark, name in (("neat-genai-studio", "Neat GenAI Studio"), ("webapp/server.py", "Neat Decision Studio (another copy)"),
                       ("/laya ", "The Laya Runtime (run by hand)"), ("llima", "LLiMa"), ("polima", "PoLiMa")):
        if mark in command:
            return name
    words = command.split()
    return Path(words[0]).name if words else "an unnamed program"


def studio_memory(url: str) -> dict | None:
    """What NEAT GenAI Studio says it has on the MLA: {"models": {name: bytes}, "bytes"}.

    Its control port (9997, on the board only) reports the models it has loaded with sizes
    estimated from their files. It is its own bookkeeping: after somebody else has reset the
    accelerator it still lists models that are no longer there, until it loads them again.
    """
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/control/memory", timeout=1.5) as response:
            body = json.loads(response.read())
        models = {str(name): int(size) for name, size in (body.get("models") or {}).items()}
        return {"models": models, "bytes": sum(models.values()), "estimated": bool(body.get("estimated", True))}
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def reset_accelerator() -> str:
    """Restart the board's MLA services, which unloads every model on the MLA: this app's and
    every other application's. It is what clears a failed load that left memory held.

    The command is the one `./run.sh --reset-mla` runs, and it needs root: without a password
    if sudo allows that, otherwise with MLA_SUDO_PASSWORD (run.sh passes the DevKit image's).
    Returns the end of the command's output; raises Refused if it could not be run.
    """
    command = os.environ.get("LAYA_RESET_COMMAND")              # for trying this off the board
    if command:
        argv = ["sh", "-c", command]
    elif os.access("/usr/bin/fix_devkit_runtime.sh", os.X_OK):
        argv = ["sudo", "-n", "/usr/bin/fix_devkit_runtime.sh"]
    else:
        argv = ["sudo", "-n", "systemctl", "restart", "simaai-appcomplex.service"]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=180)
        password = os.environ.get("MLA_SUDO_PASSWORD")
        if done.returncode != 0 and argv[0] == "sudo" and password:
            done = subprocess.run(["sudo", "-S", "-p", ""] + argv[2:], input=password + "\n", capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Refused(f"the reset could not be run: {error}") from error
    tail = " | ".join((done.stdout + done.stderr).strip().splitlines()[-3:])
    if done.returncode != 0:
        raise Refused(f"the reset failed ({done.returncode}): {tail or 'no output'}")
    time.sleep(3)                           # the services need a moment before a model can be loaded again
    return tail


class ModelManager:
    """Loads and unloads models on request, keeping at most one on the MLA.

    Loading a model unloads the one that was there. Nothing else is loaded or unloaded behind
    the user's back: a request for a model that is not loaded is refused, not served by a
    switch.
    """

    # Linux RAM kept free after a load, on top of the embedding table and tokenizer it brings.
    HEADROOM = 600 << 20

    def __init__(self, models: dict[str, Model]):
        self.models = models
        self._admin = threading.Lock()   # one load or unload at a time
        self._mla_total = mla_memory_total()
        self._held = MlaHeld()
        self._order: list[str] = []      # loaded models, oldest first
        self.studio_control: str | None = None   # NEAT GenAI Studio's control port, to ask what it has loaded
        self._report_lock = threading.Lock()
        self._report_at, self._report = 0.0, None
        self.resetting = False

    def mla(self) -> dict:
        """The MLA's memory in full, for Settings: what is held, by this app, and by whom else.

        `held_bytes` is measured (the dispatcher's map) and is what decides whether a model
        fits. The shares are not measured: this app's is the size of its graph files, and
        another program's is what that program reports, where it has a way to be asked. So
        `other_bytes`, what is held beyond this app's graphs, is the figure to go by for
        everybody else together; the programs listed say who they are.
        """
        with self._report_lock:
            if time.monotonic() - self._report_at < 3.0 and self._report is not None:
                return self._report
            held = self._held.read()
            # This app's share: what the MLA took on as each model was loaded, where that was
            # measured; else the size of the graph files, which runs a little high.
            loaded = [model for model in self.models.values() if model.state == "loaded"]
            measured = all(model.held_bytes is not None for model in loaded)
            mine = sum(model.held_bytes for model in loaded) if measured else self._loaded_bytes()[0]
            if held is not None:
                mine = min(mine, held)
            own = {model.process.pid for model in self.models.values() if model.process is not None and model.process.pid}
            programs = self._held.clients(own)
            for program in programs:
                if program["name"] == "Neat GenAI Studio" and self.studio_control:
                    program["reported"] = studio_memory(self.studio_control)
            self._report_at = time.monotonic()
            self._report = {"total_bytes": self._mla_total, "held_bytes": held, "app_bytes": mine, "app_measured": measured,
                            "other_bytes": None if held is None else max(held - mine, 0), "programs": programs}
            return self._report

    def loaded(self) -> list[str]:
        """The models on the MLA, in the order they were loaded."""
        return [name for name in self._order if name in self.models and self.models[name].state == "loaded"]

    def describe(self) -> dict:
        total = int(Path("/proc/meminfo").read_text().split()[1]) * 1024
        arrived, arriving = self._loaded_bytes()
        return {"version": VERSION, "loaded": self.loaded(),
                "models": {name: model.describe() for name, model in self.models.items()},
                "memory": {"available_bytes": memory_available(), "total_bytes": total,
                           "mla_total_bytes": self._mla_total, "mla_loaded_bytes": arrived, "mla_arriving_bytes": arriving,
                           "mla_held_bytes": self._held.read()},
                "resetting": self.resetting}

    def _loaded_bytes(self) -> tuple[int, int]:
        """Graph bytes this app has on the MLA at this moment, and of a graph still arriving.

        This is the app's own count, kept as the runtime reports each graph. What every
        program on the board holds together is `MlaHeld`.
        """
        parts = [model.mla_bytes() for model in self.models.values()]
        return sum(part[0] for part in parts), sum(part[1] for part in parts)

    def load(self, name: str, seq_lens: list[int] | None = None):
        model = self.models[name]
        wanted = sorted(set(seq_lens)) if seq_lens else list(model.seq_lens)
        if not wanted or any(s not in model.seq_lens for s in wanted):
            raise Refused(f"{name} has graphs for {model.seq_lens} tokens, not {wanted}")
        with self._admin:
            if model.state == "loaded" and model.loaded_seq_lens == wanted:
                return
            # A load that cannot be satisfied fails halfway and leaves memory held by the MLA
            # server until the MLA services are reset, so refuse what plainly cannot fit.
            graphs = sum(model.graph_bytes(s) for s in wanted)
            if self._mla_total and graphs > self._mla_total:
                raise Refused(
                    f"{name} does not fit: its graphs take {graphs >> 20} MB and the MLA has "
                    f"{self._mla_total >> 20} MB. Load fewer graphs.")
            # Nor what does not fit beside what every program on the board has taken, this
            # app's other models included: they stay. Only this model's own graphs, if it is
            # loaded already with others, go first and are counted as coming free, though the
            # dispatcher does not always give memory back.
            held = before = self._held.read(fresh=True)
            if held is None:                      # no reading: go by this app's own count
                held = sum(other.mla_bytes()[0] for other in self.models.values())
            was = model.held_bytes or 0           # what it held, if it is being loaded again
            if self._mla_total:
                free = self._mla_total - held + model.mla_bytes()[0]
                if graphs > free:
                    raise Refused(
                        f"{name} does not fit right now: its graphs take {graphs / (1 << 30):.1f} GB and "
                        f"{max(free, 0) / (1 << 30):.1f} GB of the MLA's {self._mla_total / (1 << 30):.0f} GB is free. "
                        f"The rest is held for every program on the board: models loaded here or "
                        f"elsewhere, and memory not given back yet. Unload a model, or Reset "
                        f"Accelerator, which frees it all and unloads every program's models.")
            if memory_available() < model.fixed_bytes() + self.HEADROOM:
                raise Refused(
                    f"not enough board memory for {name}'s embedding table "
                    f"({memory_available() >> 20} MB available).")
            # Other models stay where they are. This one goes first if it is loaded already,
            # with other graphs.
            others = [model] if model.process is not None else []
            progress = Progress(model.loading_stages(wanted, [other.name for other in others]))
            model.state, model.progress = "loading", progress
            process = LayaProcess(model._laya, str(model.path),
                                  ",".join(map(str, wanted)) if model.kind == "laya" else None)
            first_graph = len(others) + 1           # stages: unloads, reading, graph files, warm-up
            elfs = [file for s in wanted for file in model.graph_files(s)]

            def follow(line: str):
                """The runtime says when it starts and finishes each graph file."""
                for position, elf in enumerate(elfs):
                    if line.endswith(elf) and line.startswith("Loading model"):
                        progress.enter(first_graph + position)
                        model.arriving = (elf, time.monotonic(), seconds_to_load_graph(model.file_bytes(elf)))
                    elif line.endswith(elf) and line.startswith("Done loading"):
                        progress.enter(first_graph + position + 1)
                        model.arriving = None
                        if elf not in model.arrived_files:
                            model.arrived_files = model.arrived_files + [elf]
                        model.resident = [s for s in wanted
                                          if all(file in model.arrived_files for file in model.graph_files(s))]

            try:
                for position, other in enumerate(others):
                    progress.enter(position)
                    self._unload(other, show=other is not model)
                model.state, model.progress = "loading", progress
                progress.enter(len(others))
                process.start(follow)
            except (RuntimeDied, OSError):
                model.state, model.progress = "unloaded", None
                model.resident, model.arrived_files, model.arriving = [], [], None
                raise
            model.process, model.loaded_seq_lens = process, wanted
            model.resident, model.arriving = list(wanted), None
            model.arrived_files = elfs
            after = self._held.read(fresh=True)
            model.held_bytes = None if before is None or after is None else max(was + after - before, 0)
            self._order = [other for other in self._order if other != name] + [name]
            model.state, model.progress = "loaded", None

    def _unload(self, model: Model, show: bool = True):
        if model.process is not None:
            if show:
                model.state, model.progress = "unloading", Progress([("Unloading", SECONDS_TO_UNLOAD)])
            model.process.stop()   # waits for a request in flight
        model.process, model.loaded_seq_lens = None, []
        model.resident, model.arrived_files, model.arriving = [], [], None
        model.held_bytes = None
        model.state, model.progress = "unloaded", None

    def unload(self, name: str):
        with self._admin:
            self._unload(self.models[name])

    def unload_all(self):
        with self._admin:
            for model in self.models.values():
                self._unload(model)

    def ask_all(self, names: list[str], payloads: dict[str, dict]) -> dict[str, dict]:
        """The same question to several loaded models at once, each in its own thread.

        Each model has its own runtime process; the MLA takes their graphs in turn. A model
        that fails answers {"error": ...} and the others still answer.
        """
        results: dict[str, dict] = {}

        def work(name: str):
            start = time.perf_counter()
            try:
                answer = self.request(name, payloads[name])
            except (NotLoaded, Refused, RuntimeDied, OSError) as error:
                answer = {"error": str(error)}
            answer["server_ms"] = round((time.perf_counter() - start) * 1e3, 3)
            results[name] = answer

        threads = [threading.Thread(target=work, args=(name,), daemon=True) for name in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return {name: results[name] for name in names}

    def request(self, name: str, payload: dict) -> dict:
        model = self.models[name]
        process = model.process
        if model.state != "loaded" or process is None:
            raise NotLoaded(f"the {name} model is not loaded")
        if not process.running:
            self.unload(name)
            raise RuntimeDied(f"the {name} model's runtime has stopped; load it again")
        return process.request(payload)

    def reset(self) -> str:
        """Unload this app's model, then reset the accelerator for everybody."""
        with self._admin:
            self.resetting = True
            try:
                for model in self.models.values():
                    self._unload(model)
                return reset_accelerator()
            finally:
                self.resetting = False

    def add(self, model: Model):
        """A model that has just arrived on the board. Readers keep the mapping they have."""
        self.models = {**self.models, model.name: model}

    def delete(self, name: str):
        """Remove an unloaded model's files from the board's disk.

        Only the files its configuration names are removed, the configuration first, so that
        a directory is never left looking complete; the directory goes if that empties it.
        """
        with self._admin:
            model = self.models[name]
            if model.state != "unloaded" or model.process is not None:
                raise Refused(f"{name} is on the MLA; unload it before deleting it")
            names = [CONFIGS[model.kind], *(file for files in model._files.values() for file in files),
                     *model.other_files()]
            if any(Path(file).name != file for file in names):
                raise Refused(f"{name}'s configuration names files outside its directory; not deleting")
            # Out of the list first: a page asking about it meanwhile would look for its files.
            self.models = {other: kept for other, kept in self.models.items() if other != name}
            for file in names:
                (model.path / file).unlink(missing_ok=True)
            try:
                model.path.rmdir()
            except OSError:
                pass                              # something else is in there: leave it

    def stop(self):
        for model in self.models.values():
            self._unload(model)


def yes_no(kind: str, text: str) -> tuple[str, dict]:
    """A bare yes-or-no question as each kind of model is best asked it: (state, question).

    Laya answers best with the text as the state, labelled a question, and the two options
    (of eight ways of putting it, measured for the Debate page). CLM is asked the way its
    heads were trained: the question itself, with nothing before it.
    """
    if kind == "clm":
        return "", {"type": "noul", "instructions": text}
    return "Question: " + text, {"type": "choice", "instructions": "Is the answer yes or no?", "criteria": ["yes", "no"]}


def yes_share(answer: dict) -> float | None:
    """The probability of yes in an answer to `yes_no`."""
    if answer.get("type") == "noul":
        return answer.get("noul")
    return (answer.get("probabilities") or {}).get("yes")


class Cancelled(RuntimeError):
    pass


class Hub:
    """Compiled models published on Hugging Face, and fetching one onto this board.

    The repository's `models.json` (written by tools/publish_hub.py) lists each model's files
    with sizes and SHA-256 sums. A download writes them into the app directory, as `model`
    for the general model and `model-<name>` otherwise, which is where run.sh looks at
    startup; once complete the model is handed to the manager and can be loaded. What comes
    from the network is checked: names against a pattern, every file against its sum.
    """

    REFRESH = 300                    # seconds a fetched list is good for
    NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,40}")
    FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}")
    SPARE = 1 << 30                  # disk left free after a download

    def __init__(self, repo: str, root: Path, manager: ModelManager, laya: str, allowed: str | None):
        self.repo, self.root = repo, root
        self._manager, self._laya, self._allowed = manager, laya, allowed
        self._lock = threading.Lock()
        self._models: dict | None = None     # the list, once fetched
        self._fetched = 0.0
        self._fetching = False
        self._error: str | None = None
        self._job: dict | None = None        # the download in progress, or the last one
        self._failed: dict[str, str] = {}    # model -> why its last download stopped
        self.quiet = False                   # no log lines: the caller reports (`--fetch`)

    def _url(self, path: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{path}"

    def directory(self, name: str) -> Path:
        return self.root / ("model" if name == "general" else f"model-{name}")

    def _refresh(self):
        try:
            request = urllib.request.Request(self._url("models.json"), headers={"User-Agent": "neat-decision-studio"})
            with urllib.request.urlopen(request, timeout=15) as response:
                listed = json.loads(response.read(1 << 20))["models"]
            models = {}
            for name, entry in listed.items():
                files = entry["files"]
                if not (self.NAME.fullmatch(name) and self.NAME.fullmatch(entry["path"])
                        and all(self.FILE.fullmatch(file["name"]) and isinstance(file["bytes"], int)
                                and re.fullmatch(r"[0-9a-f]{64}", file["sha256"]) for file in files)
                        and any(file["name"] == "laya_config.json" for file in files)):
                    raise ValueError(f"the entry for {name!r} is not one this app understands")
                models[name] = entry
            self._models, self._error = models, None
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._error = f"could not read the model list from Hugging Face: {error}"
        finally:
            self._fetched, self._fetching = time.monotonic(), False

    def _models_now(self) -> dict:
        """The list as last fetched; a fetch is started when there is none or it is old."""
        with self._lock:
            old = time.monotonic() - self._fetched > (self.REFRESH if self._models is not None else 15)
            if old and not self._fetching:
                self._fetching = True
                threading.Thread(target=self._refresh, daemon=True).start()
        return self._models or {}

    def _have(self, name: str, entry: dict) -> int:
        """Bytes of the model's files that are complete on the board."""
        directory = self.directory(name)
        return sum(file["bytes"] for file in entry["files"]
                   if (path := directory / file["name"]).is_file() and path.stat().st_size == file["bytes"])

    def describe(self) -> dict:
        models, job = {}, self._job
        for name, entry in self._models_now().items():
            total = sum(file["bytes"] for file in entry["files"])
            sizes = {file["name"]: file["bytes"] for file in entry["files"]}
            downloading = job is not None and job["name"] == name and job["thread"].is_alive()
            progress = None
            if downloading:
                seconds = time.monotonic() - job["started"]
                rate = job["fetched"] / seconds if seconds > 1 else 0
                progress = {"fraction": round(job["done"] / total, 4), "done_bytes": job["done"], "total_bytes": total,
                            "file": job["file"], "bytes_per_second": int(rate),
                            "seconds_left": int((total - job["done"]) / rate) if rate else None}
            models[name] = {
                "title": entry.get("title", name), "about": entry.get("about", ""),
                "precision": entry.get("precision"), "agreement": entry.get("agreement"),
                "path": entry["path"],
                "card": {key: value for key, value in entry.get("card", {}).items() if isinstance(value, str)},
                "bytes": total, "have_bytes": 0 if downloading else self._have(name, entry),
                "graphs": [{"seq_len": int(s), "bytes": sizes.get(elf, 0), "latency_ms": entry.get("latency_ms", {}).get(s)}
                           for s, elf in sorted(entry.get("graphs", {}).items(), key=lambda item: int(item[0]))],
                "state": "downloading" if downloading else "on_board" if name in self._manager.models else "available",
                "progress": progress, "error": self._failed.get(name),
            }
        return {"repo": self.repo, "page": f"https://huggingface.co/{self.repo}", "error": self._error,
                "listed": self._models is not None, "models": models,
                "free_bytes": shutil.disk_usage(self.root).free}

    def download(self, name: str):
        entry = self._models_now().get(name)
        if entry is None:
            raise Refused(f"Hugging Face lists no model named {name!r}")
        with self._lock:
            if self._job is not None and self._job["thread"].is_alive():
                raise Refused(f"{self._job['name']} is being downloaded; one download at a time")
            if name in self._manager.models:
                raise Refused(f"{name} is already on this board")
            need = sum(file["bytes"] for file in entry["files"]) - self._have(name, entry)
            free = shutil.disk_usage(self.root).free
            if free < need + self.SPARE:
                raise Refused(f"not enough disk for {name}: {need >> 20} MB to fetch, {free >> 20} MB free")
            self._failed.pop(name, None)
            job = {"name": name, "done": 0, "fetched": 0, "file": "", "started": time.monotonic(),
                   "cancel": threading.Event()}
            job["thread"] = threading.Thread(target=self._run, args=(job, entry), daemon=True)
            self._job = job
            job["thread"].start()

    def cancel(self, name: str):
        job = self._job
        if job is not None and job["name"] == name:
            job["cancel"].set()

    def discard(self, name: str):
        """Remove what a stopped download left behind."""
        entry = self._models_now().get(name)
        if entry is None:
            raise Refused(f"Hugging Face lists no model named {name!r}")
        with self._lock:
            if self._job is not None and self._job["name"] == name and self._job["thread"].is_alive():
                raise Refused(f"{name} is being downloaded; cancel that first")
            if name in self._manager.models:
                raise Refused(f"{name} is a model on this board, not a stopped download")
            directory = self.directory(name)
            for file in entry["files"]:
                for path in (directory / file["name"], directory / (file["name"] + ".part")):
                    path.unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                pass
            self._failed.pop(name, None)

    def _run(self, job: dict, entry: dict):
        name, directory = job["name"], self.directory(job["name"])
        try:
            directory.mkdir(exist_ok=True)
            # laya_config.json goes last: a directory that has it is complete, for run.sh too.
            for file in sorted(entry["files"], key=lambda file: file["name"] == "laya_config.json"):
                target = directory / file["name"]
                if target.is_file() and target.stat().st_size == file["bytes"]:
                    job["done"] += file["bytes"]
                    continue
                job["file"] = file["name"]
                self._fetch(job, f"{entry['path']}/{file['name']}", target, file)
            model = Model(name, str(directory), self._laya, None if name == "dino" else self._allowed)
            self._manager.add(model)
            if not self.quiet:
                print(f"{time.strftime('%H:%M:%S')} downloaded {name} from {self.repo}", flush=True)
        except Cancelled:
            self._failed[name] = "The download was cancelled. What had arrived is kept, and the next one continues from there."
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self._failed[name] = f"The download stopped: {error}"
            print(f"{time.strftime('%H:%M:%S')} download of {name} failed: {error}", flush=True)

    def _fetch(self, job: dict, path: str, target: Path, file: dict):
        """One file, to `<name>.part` and then into place once its size and sum are right."""
        part = target.with_name(target.name + ".part")
        digest, size = hashlib.sha256(), 0
        request = urllib.request.Request(self._url(path), headers={"User-Agent": "neat-decision-studio"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
                unsynced = 0
                while chunk := response.read(1 << 20):
                    if job["cancel"].is_set():
                        raise Cancelled()
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    job["done"] += len(chunk)
                    job["fetched"] += len(chunk)
                    unsynced += len(chunk)
                    if unsynced >= 64 << 20:
                        self._settle(out)
                        unsynced = 0
                self._settle(out)
            if size != file["bytes"] or digest.hexdigest() != file["sha256"]:
                raise ValueError(f"{file['name']} arrived damaged ({size} of {file['bytes']} bytes)")
            os.replace(part, target)
        except BaseException:
            job["done"] -= size
            part.unlink(missing_ok=True)
            raise

    @staticmethod
    def _settle(out):
        """Write out what was received and drop it from the page cache.

        Page cache takes room in the kernel's CMA pool, which is where the MLA's graphs are
        loaded, and a pool full of it makes the next model load fail.
        """
        out.flush()
        os.fsync(out.fileno())
        os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


# Snake is the game the Games tab opens on.
class LLM:
    """A language model to play against: an OpenAI-compatible chat endpoint, which is what
    NEAT GenAI Studio serves on the board (`neat-ai`, port 9998). The games ask it for a move
    in words; this only passes the question on and brings the answer back, since a page
    cannot call another port itself."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._seen, self._status = 0.0, None

    def _call(self, path: str, payload: dict | None, timeout: float) -> dict:
        request = urllib.request.Request(self.url + path, data=json.dumps(payload).encode() if payload is not None else None,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def describe(self) -> dict:
        """Whether a model is being served, and which. Asked again every few seconds at most."""
        if self._status is None or time.monotonic() - self._seen > 3:
            try:
                # A server may list its speech and embedding models next to the chat ones.
                models = [entry["id"] for entry in self._call("/v1/models", None, 2).get("data", [])
                          if not re.search(r"whisper|asr|tts|embed|gte-", entry["id"], re.I)]
                self._status = {"available": bool(models), "models": models, "url": self.url,
                                "error": None if models else "the server is up, but no chat model is loaded in it"}
            except (OSError, ValueError, KeyError, TypeError) as error:
                self._status = {"available": False, "models": [], "url": self.url, "error": f"nothing answers at {self.url} ({error})"}
            self._seen = time.monotonic()
        return self._status

    def chat(self, messages: list, max_tokens: int) -> dict:
        status = self.describe()
        if not status["available"]:
            raise Refused("no language model is being served: " + status["error"])
        model, start = status["models"][0], time.perf_counter()
        try:
            body = self._call("/v1/chat/completions", {
                "model": model, "messages": messages, "max_tokens": max_tokens, "stream": False, "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False}}, 180)
            text = body["choices"][0]["message"]["content"] or ""
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            self._status = None
            raise Refused(f"the language model did not answer: {error}") from error
        # A reasoning model's thinking is not its answer.
        text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()
        return {"text": text, "model": model, "ms": round((time.perf_counter() - start) * 1e3, 1),
                "tokens": (body.get("usage") or {}).get("completion_tokens")}


PAGES = {"/": "landing.html", "/showcase": "showcase.html", "/debate": "debate.html", "/questions": "index.html", "/games": "snake.html",
         "/games/snake": "snake.html", "/games/chess": "chess.html", "/games/tictactoe": "tictactoe.html",
         "/games/rps": "rps.html", "/games/dino": "games.html", "/games/blackjack": "blackjack.html",
         "/games/sudoku": "sudoku.html", "/compare": "compare.html",
         "/models": "landing.html"}        # the landing page, on which settings.js opens Settings
# Scripts, the shared stylesheet, and the brand images and fonts it uses.
ASSET = re.compile(r"/static/((?:brand/|fonts/|vendor/)?[a-z][a-z0-9-]*\.(js|css|png|svg|woff2))")
ASSET_TYPES = {"js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8", "png": "image/png",
               "svg": "image/svg+xml", "woff2": "font/woff2"}


stopping = threading.Event()             # set when the app is shutting down, to end the pages' event streams


def reset_token(root: Path) -> str:
    """The word a browser on another machine has to give to reset the accelerator.

    Resetting unloads other applications' models too, so it is not open to whoever can reach
    the port: a request from the board itself needs nothing, any other needs this token. It is
    made once and kept in the app directory, readable by the user the app runs as.
    """
    path = root / ".reset-token"
    try:
        token = path.read_text().strip()
        if token:
            return token
    except OSError:
        pass
    token = secrets.token_hex(4)
    try:
        path.write_text(token + "\n")
        path.chmod(0o600)
    except OSError:
        pass                              # it still works for as long as this app runs
    return token


def make_handler(manager: ModelManager, hub: Hub | None, llm: LLM | None = None, token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Headers and body must leave in one segment: sent separately, Nagle's algorithm holds
        # the body until the client's delayed ACK, which adds 40 ms to a 19 ms decision.
        wbufsize = 64 * 1024

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, fmt, *args):
            pass

        def _note(self, what: str):
            """Loads and unloads are rare and change what every page sees: keep a trace."""
            print(f"{time.strftime('%H:%M:%S')} {self.client_address[0]}: {what}", flush=True)

        def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def _json(self, status: int, payload: dict):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in PAGES:
                self._send(200, (STATIC / PAGES[path]).read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/info":
                self._json(200, manager.describe())
            elif path == "/api/events":
                self._events()
            elif path == "/api/llm":
                self._json(200, llm.describe() if llm else {"available": False, "models": [], "url": None,
                                                            "error": "this app was started without a language model to play (--llm none)"})
            elif path == "/api/mla":
                self._json(200, manager.mla())
            elif path == "/api/hub":
                self._json(200, hub.describe() if hub else {"repo": None, "listed": True, "models": {}, "error": None})
            elif (asset := ASSET.fullmatch(path)) and (STATIC / asset[1]).is_file():
                # Images and fonts do not change between deploys; code does.
                self._send(200, (STATIC / asset[1]).read_bytes(), ASSET_TYPES[asset[2]],
                           cache="max-age=86400" if asset[2] in ("png", "svg", "woff2") else "no-store")
            else:
                self._json(404, {"error": "not found"})

        def _events(self):
            """What /api/info says, pushed whenever it changes (server-sent events).

            A page that listens sees a model arrive on the MLA graph by graph, and what another
            page or the terminal does, without asking. While something is loading the picture
            is taken ten times a second; otherwise twice. Linux's free memory moves all the
            time and is left out of what counts as a change.
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            last, sent = None, 0.0
            try:
                while not stopping.is_set():
                    picture = manager.describe()
                    busy = any(model["progress"] for model in picture["models"].values())
                    key = json.dumps({**picture, "memory": {k: v for k, v in picture["memory"].items() if k != "available_bytes"}})
                    if key != last or time.monotonic() - sent > 10:          # a line now and then keeps the connection alive
                        self.wfile.write(b"data: " + json.dumps(picture).encode() + b"\n\n")
                        self.wfile.flush()
                        last, sent = key, time.monotonic()
                    stopping.wait(0.1 if busy else 0.5)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass                                   # the page went away

        def _body(self, known: bool = True) -> dict | None:
            """The request's JSON object, or None after an error response has been sent.

            Its "model" has to be one on the board, unless `known` is off.
            """
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body too large"})
                    return None
                request = json.loads(self.rfile.read(length))
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "the request body is not valid JSON"})
                return None
            if not isinstance(request, dict):
                self._json(400, {"error": "expected a JSON object"})
                return None
            # No model named: the one that is loaded, or the first of several.
            name = request.get("model") or next(iter(manager.loaded()), "general")
            if not isinstance(name, str) or (known and name not in manager.models):
                self._json(404, {"error": f"there is no model named {name!r}"})
                return None
            request["model"] = name
            return request

        def _hub_post(self):
            """Download a model from Hugging Face, or stop downloading it."""
            request = self._body(known=False)
            if request is None:
                return
            if hub is None:
                return self._json(404, {"error": "this app was started without a model hub"})
            name = request["model"]
            try:
                if self.path == "/api/hub/download":
                    self._note(f"download {name}")
                    hub.download(name)
                else:
                    self._note(f"cancel the download of {name}")
                    hub.cancel(name)
            except Refused as error:
                return self._json(409, {"error": str(error)})
            self._json(200, hub.describe())

        def _reset(self):
            """Reset the accelerator: from the board freely, from elsewhere with the token."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length)) if 0 < length <= 4096 else {}
            except (ValueError, json.JSONDecodeError):
                request = {}
            local = self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")
            given = request.get("token") if isinstance(request, dict) else None
            if not local and not (token and isinstance(given, str) and secrets.compare_digest(given.strip(), token)):
                self._note("reset refused: no token, or not the right one")
                return self._json(403, {"error": "Resetting the accelerator from another machine needs the reset token.", "needs_token": True})
            self._note("reset the accelerator")
            try:
                said = manager.reset()
            except Refused as error:
                self._note(f"reset failed: {error}")
                return self._json(500, {"error": str(error)})
            self._json(200, {**manager.describe(), "reset": said or "done"})

        def _compare(self):
            """One question to several loaded models at once (default: all of them).

            Either {"yes_no": "a question"}, which each model is asked in the form that suits
            its kind and answered with "yes", the probability of yes; or {"state", "questions"}
            as for /api/predict, sent to each as it is. "models" names which.
            """
            request = self._body(known=False)
            if request is None:
                return
            names = request.get("models") or manager.loaded()
            if not isinstance(names, list) or any(name not in manager.models for name in names):
                return self._json(404, {"error": "models must be a list of models on this board"})
            if not names:
                return self._json(409, {"error": "no model is loaded", "not_loaded": True})
            text = request.get("yes_no")
            if isinstance(text, str) and text.strip():
                payloads = {}
                for name in names:
                    state, question = yes_no(manager.models[name].kind, text.strip())
                    payloads[name] = {"state": state, "questions": {"answer": question}}
            elif "state" in request and "questions" in request:
                payloads = {name: {"state": request["state"], "questions": request["questions"]} for name in names}
            else:
                return self._json(400, {"error": "expected {\"yes_no\": ...} or {\"state\": ..., \"questions\": {...}}"})
            start = time.perf_counter()
            results = manager.ask_all(names, payloads)
            if isinstance(text, str) and text.strip():
                for result in results.values():
                    if "answers" in result:
                        result["yes"] = yes_share(result["answers"]["answer"])
            self._json(200, {"results": results, "models": names,
                             "server_ms": round((time.perf_counter() - start) * 1e3, 3)})

        def _llm_chat(self):
            """A game's question to the language model it is playing."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length)) if 0 < length <= 65536 else None
            except (ValueError, json.JSONDecodeError):
                request = None
            messages = request.get("messages") if isinstance(request, dict) else None
            if not (isinstance(messages, list) and 0 < len(messages) <= 8 and all(
                    isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
                    and isinstance(m.get("content"), str) and len(m["content"]) <= 6000 for m in messages)):
                return self._json(400, {"error": "expected {\"messages\": [{\"role\": ..., \"content\": ...}]}"})
            if llm is None:
                return self._json(409, {"error": "this app was started without a language model to play (--llm none)"})
            limit = request.get("max_tokens")
            try:
                self._json(200, llm.chat([{"role": m["role"], "content": m["content"]} for m in messages],
                                         limit if isinstance(limit, int) and 0 < limit <= 128 else 24))
            except Refused as error:
                self._json(409, {"error": str(error)})

        def _delete(self):
            """Delete a model from the board's disk, or the remains of a stopped download."""
            request = self._body(known=False)
            if request is None:
                return
            name = request["model"]
            try:
                if name in manager.models:
                    self._note(f"delete {name}")
                    manager.delete(name)
                elif hub is not None:
                    self._note(f"discard the partial download of {name}")
                    hub.discard(name)
                else:
                    return self._json(404, {"error": f"there is no model named {name!r}"})
            except Refused as error:
                return self._json(409, {"error": str(error)})
            except OSError as error:
                return self._json(500, {"error": f"could not delete {name}: {error}"})
            self._json(200, manager.describe())

        def do_POST(self):
            if self.path in ("/api/hub/download", "/api/hub/cancel"):
                return self._hub_post()
            if self.path == "/api/models/delete":
                return self._delete()
            if self.path == "/api/llm/chat":
                return self._llm_chat()
            if self.path == "/api/mla/reset":
                return self._reset()
            if self.path == "/api/compare":
                return self._compare()
            if self.path not in ("/api/predict", "/api/models/load", "/api/models/unload"):
                return self._json(404, {"error": "not found"})
            request = self._body()
            if request is None:
                return
            name = request["model"]
            try:
                if self.path == "/api/models/load":
                    seq_lens = request.get("seq_lens")
                    if seq_lens is not None and not (isinstance(seq_lens, list)
                                                     and all(isinstance(s, int) for s in seq_lens)):
                        return self._json(400, {"error": "seq_lens must be a list of integers"})
                    self._note(f"load {name}")
                    manager.load(name, seq_lens)
                    return self._json(200, manager.describe())
                if self.path == "/api/models/unload":
                    if request.get("all") is True:
                        self._note("unload all")
                        manager.unload_all()
                    else:
                        self._note(f"unload {name}")
                        manager.unload(name)
                    return self._json(200, manager.describe())
                if "state" not in request or "questions" not in request:
                    return self._json(400, {"error": "expected {\"state\": ..., \"questions\": {...}}"})
                forward = {"state": request["state"], "questions": request["questions"]}
                # seq_len pins one compiled graph; max_len and head_max_len set the token budget.
                for key in ("seq_len", "max_len", "head_max_len"):
                    if isinstance(request.get(key), int) and request[key] > 0:
                        forward[key] = request[key]
                start = time.perf_counter()
                response = manager.request(name, forward)
            except NotLoaded as error:
                return self._json(409, {"error": str(error), "not_loaded": name})
            except Refused as error:
                self._note(f"refused: {error}")
                return self._json(409, {"error": str(error)})
            except (RuntimeDied, OSError) as error:
                self._note(f"failed: {error}")
                return self._json(503, {"error": str(error)})
            response["server_ms"] = round((time.perf_counter() - start) * 1e3, 3)
            self._json(400 if "error" in response else 200, response)

    return Handler


def load_shown(manager: ModelManager, name: str):
    """Load a model, drawing its progress in the terminal: a spinner, a bar in the Neat
    colours and the stage it is at. The stages are the ones the Models page shows. Without a
    terminal the load is silent, as a log should be."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return manager.load(name)
    failure = []

    def work():
        try:
            manager.load(name)
        except (RuntimeDied, Refused, OSError) as error:
            failure.append(error)

    thread = threading.Thread(target=work, daemon=True)
    started = time.monotonic()
    thread.start()
    palette = [(22, 200, 166), (75, 181, 74), (169, 200, 28), (58, 134, 236), (242, 128, 29)]
    spinner, width, tick = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", 30, 0

    def colour(position: float) -> str:               # the spectrum, from teal to orange along the bar
        at = position * (len(palette) - 1)
        low, mix = min(int(at), len(palette) - 2), at - min(int(at), len(palette) - 2)
        r, g, b = (round(palette[low][k] + (palette[low + 1][k] - palette[low][k]) * mix) for k in range(3))
        return f"\033[38;2;{r};{g};{b}m"

    while thread.is_alive():
        progress = manager.models[name].progress
        shown = progress.describe() if progress else {"fraction": 0.0, "stage": "Starting", "step": 1, "steps": 1}
        filled = round(shown["fraction"] * width)
        bar = "".join(colour(k / (width - 1)) + "█" for k in range(filled)) + "\033[38;2;60;66;74m" + "░" * (width - filled) + "\033[0m"
        sys.stdout.write(f"\r\033[K   \033[38;2;47;212;192m{spinner[tick % len(spinner)]}\033[0m {bar} {round(shown['fraction'] * 100):3d}%  "
                         f"\033[38;2;140;150;160m{shown['stage']}\033[0m")
        sys.stdout.flush()
        tick += 1
        thread.join(0.08)
    sys.stdout.write("\r\033[K")
    if failure:
        raise failure[0]
    graphs = ", ".join(map(str, manager.models[name].loaded_seq_lens))
    print(f"   \033[38;2;53;196;137m✔\033[0m {name} is on the MLA ({graphs}-token graphs) "
          f"\033[38;2;140;150;160min {time.monotonic() - started:.1f} s\033[0m", flush=True)


def hub_command(hub: Hub | None, manager: ModelManager, args) -> int:
    """`--list-hub` and `--fetch`: the hub from a terminal, as setup.sh uses it. Nothing is served
    and the MLA is not touched."""
    if hub is None:
        print("laya webapp: started with --hub none; there is nothing to list or fetch", file=sys.stderr)
        return 1
    hub.quiet = True
    hub._refresh()
    listed = hub.describe()
    if listed["error"]:
        print(f"laya webapp: {listed['error']}", file=sys.stderr)
        return 1
    megabytes = lambda count: f"{count / 1e9:.2f} GB" if count >= 1e9 else f"{count / 1e6:.0f} MB"
    if args.list_hub:
        for name, model in listed["models"].items():
            print(f"{name}\t{model['title']}\t{megabytes(model['bytes'])}\t{model['state']}")
        return 0
    names = list(listed["models"]) if args.fetch == ["all"] else args.fetch
    failed = 0
    for name in names:
        if name in manager.models:
            print(f"  {name}: already on this board")
            continue
        try:
            hub.download(name)
        except Refused as error:
            print(f"  {name}: {error}", file=sys.stderr)
            failed += 1
            continue
        job, shown = hub._job, -1
        try:
            while job["thread"].is_alive():
                progress = hub.describe()["models"][name]["progress"]
                if progress and sys.stdout.isatty():
                    print(f"\r\033[K  {name}: {megabytes(progress['done_bytes'])} of {megabytes(progress['total_bytes'])}"
                          f" · {progress['fraction'] * 100:.0f}% · {progress['bytes_per_second'] / 1e6:.1f} MB/s", end="", flush=True)
                elif progress and int(progress["fraction"] * 10) > shown:      # a log: a line every tenth
                    shown = int(progress["fraction"] * 10)
                    print(f"  {name}: {progress['fraction'] * 100:.0f}% of {megabytes(progress['total_bytes'])}", flush=True)
                job["thread"].join(0.5)
        except KeyboardInterrupt:
            hub.cancel(name)
            job["thread"].join()
            print(f"\n  {name}: cancelled; what had arrived is kept")
            return 130
        if sys.stdout.isatty():
            print("\r\033[K", end="")
        if name in manager.models:
            print(f"  {name}: downloaded")
        else:
            print(f"  {name}: {hub._failed.get(name, 'the download stopped')}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/media/nvme/laya/model", help="compiled model directory")
    ap.add_argument("--laya", default="/media/nvme/laya/laya", help="the laya runtime binary")
    ap.add_argument("--seq-lens", help="offer only these compiled sequence lengths, e.g. 128,512")
    ap.add_argument("--game-model", help="compiled model directory for the games page")
    ap.add_argument("--extra-model", action="append", default=[], metavar="NAME=DIR",
                    help="another question model (repeatable)")
    ap.add_argument("--preload", default="general", metavar="NAME",
                    help="the model to load at startup, several separated by commas, or 'none' "
                         "(default: %(default)s). More are loaded, beside it, in Settings.")
    ap.add_argument("--hub", default="TDoSiMa/sima-laya", metavar="REPO",
                    help="Hugging Face repository the Models page offers downloads from, or 'none' "
                         "(default: %(default)s)")
    ap.add_argument("--root", help="where downloaded models go: `model` and `model-<name>` in this "
                                   "directory (default: the directory holding --model)")
    ap.add_argument("--llm", default="http://127.0.0.1:9998", metavar="URL",
                    help="an OpenAI-compatible chat server the games can play against, such as NEAT GenAI "
                         "Studio's on this board, or 'none' (default: %(default)s)")
    ap.add_argument("--studio-control", default="", metavar="URL",
                    help="NEAT GenAI Studio's control port, asked which models it has on the MLA for the "
                         "memory report in Settings, or 'none' (default: port 9997 beside --llm)")
    ap.add_argument("--list-hub", action="store_true",
                    help="print the models on Hugging Face (name, title, size, state; tab-separated) and exit")
    ap.add_argument("--fetch", nargs="+", metavar="NAME",
                    help="download these models from Hugging Face (or 'all') and exit, without serving")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8095)
    args = ap.parse_args()

    # A board may have no model yet: the Models page can then fetch one.
    sources = {}
    if model_kind(args.model):
        sources["general"] = (args.model, args.seq_lens)
    for item in args.extra_model:
        name, _, path = item.partition("=")
        if not name or not path or name in ("general", "dino"):
            sys.exit(f"laya webapp: --extra-model needs NAME=DIR with a new name, got {item!r}")
        sources[name] = (path, args.seq_lens)
    if args.game_model:
        sources["dino"] = (args.game_model, None)
    try:
        manager = ModelManager({name: Model(name, path, args.laya, allowed)
                                for name, (path, allowed) in sources.items()})
    except (OSError, KeyError, json.JSONDecodeError) as error:
        sys.exit(f"laya webapp: cannot read a model directory: {error}")

    root = Path(args.root) if args.root else Path(args.model).resolve().parent
    hub = None if args.hub == "none" else Hub(args.hub, root, manager, args.laya, args.seq_lens)

    if args.list_hub or args.fetch:
        sys.exit(hub_command(hub, manager, args))

    preload = [] if args.preload == "none" else [name for name in args.preload.split(",") if name]
    if preload == ["general"] and "general" not in manager.models:
        print("laya webapp: the general model is not on this board; starting with nothing loaded", flush=True)
        preload = []
    for name in preload:
        if name not in manager.models:
            sys.exit(f"laya webapp: --preload names {name!r}, which is not one of {list(sources)}")
        try:
            load_shown(manager, name)
        except (RuntimeDied, Refused, OSError) as error:
            manager.stop()
            sys.exit(f"laya webapp: cannot load the {name} model: {error}")
    llm = None if args.llm == "none" else LLM(args.llm)
    # NEAT GenAI Studio's control port is next to its chat port, on the board only.
    if args.studio_control != "none":
        manager.studio_control = args.studio_control or (
            re.sub(r":\d+$", ":9997", args.llm.rstrip("/")) if args.llm != "none" and re.search(r":\d+/?$", args.llm) else "http://127.0.0.1:9997")
    token = reset_token(root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager, hub, llm, token))
    server.daemon_threads = True          # an open event stream must not hold up shutting down
    server.block_on_close = False

    def on_sigterm(*_):
        # Leave through the same path as Ctrl-C, so the runtimes release their models on the MLA.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    print(f"Laya playground on http://{args.host}:{args.port}  models: {', '.join(sources) or 'none'}; "
          f"loaded: {', '.join(preload) or 'none'}; hub: {args.hub}", flush=True)
    print(f"Resetting the accelerator from another machine's browser needs this token (kept in {root / '.reset-token'}): {token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        server.server_close()
        manager.stop()


if __name__ == "__main__":
    main()
