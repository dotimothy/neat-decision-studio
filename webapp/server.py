#!/usr/bin/env python3
"""Laya decision playground: a web front end for the Laya runtime on a Modalix DevKit.

Runs on the board with nothing but the Python standard library. It keeps a `laya serve`
process alive for the loaded model, so its compiled graphs stay on the MLA, and forwards each
browser request to it as one JSON line.

    /           Debate, the default: a yes-or-no question answered live as it is typed
    /questions  question answering with your own questions and options
    /games      games a model plays: Dino Arena, Blackjack, Snake, Sudoku
    /models     the model manager: every model on the board or on Hugging Face, with
                download, load, unload and delete

One model is on the MLA at a time. Models are loaded only on request, on the Models page, and
loading one unloads whichever was there; `--preload` chooses what is loaded at startup. The
app also starts with no model on the board at all: the Models page then offers the ones
published on Hugging Face (`--hub`), downloads one into the app directory and lists it.

    python3 webapp/server.py --model /media/nvme/laya/model --laya /media/nvme/laya/laya
"""
import argparse
import hashlib
import json
import os
import re
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


class RuntimeDied(RuntimeError):
    pass


class LayaProcess:
    """One `laya serve` child. The MLA runs one graph at a time, so requests are serialized."""

    def __init__(self, laya: str, model: str, seq_lens: str | None):
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


class Model:
    """One compiled model directory and, while it is loaded, the runtime process holding it."""

    def __init__(self, name: str, path: str, laya: str, allowed: str | None):
        self.name, self.path, self._laya = name, Path(path), laya
        self.config = json.loads((self.path / "laya_config.json").read_text())
        sizes = sorted(int(s) for s in self.config["elfs"])
        if allowed:
            wanted = {int(s) for s in allowed.split(",")}
            sizes = [s for s in sizes if s in wanted]
        self.seq_lens = sizes                 # graphs this app may load
        self.loaded_seq_lens: list[int] = []
        self.state = "unloaded"               # unloaded | loading | loaded | unloading
        self.progress: Progress | None = None  # while loading or unloading
        self.process: LayaProcess | None = None

    def graph_bytes(self, seq_len: int) -> int:
        return (self.path / self.config["elfs"][str(seq_len)]).stat().st_size

    def fixed_bytes(self) -> int:
        """What the runtime holds besides graphs: the embedding table and the tokenizer."""
        return sum((self.path / self.config[key]).stat().st_size for key in ("token_embeddings", "tokenizer"))

    def describe(self) -> dict:
        return {
            "checkpoint": self.config.get("model"), "precision": self.config.get("precision"),
            "max_len": self.config.get("max_len"), "hidden_size": self.config.get("hidden_size"),
            "graphs": [{"seq_len": s, "bytes": self.graph_bytes(s)} for s in self.seq_lens],
            "seq_lens": self.seq_lens, "fixed_bytes": self.fixed_bytes(),
            "state": self.state, "loaded": self.state == "loaded",
            "loaded_seq_lens": self.loaded_seq_lens,
            "progress": progress.describe() if (progress := self.progress) else None,
        }

    def loading_stages(self, seq_lens: list[int], unloading: list[str]) -> list[tuple[str, float]]:
        """What loading these graphs goes through, in order, with estimates."""
        fixed = seconds_to_read((self.path / self.config["tokenizer"]).stat().st_size,
                                (self.path / self.config["token_embeddings"]).stat().st_size)
        return ([(f"Unloading {other}", SECONDS_TO_UNLOAD) for other in unloading]
                + [("Reading the tokenizer and embeddings", fixed)]
                + [(f"Loading the {s}-token graph", seconds_to_load_graph(self.graph_bytes(s))) for s in seq_lens]
                + [("Warming up", SECONDS_TO_WARM_UP)])


def memory_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def mla_memory_total() -> int:
    """Size of the memory region reserved for MLA models (the device tree's `dms` node).

    Compiled graphs are loaded there, not into Linux RAM, so loading one barely moves
    MemAvailable. The region is shared by every application that uses the MLA.
    """
    for node in Path("/sys/firmware/devicetree/base/reserved-memory").glob("dms@*"):
        try:
            reg = (node / "reg").read_bytes()
            return int.from_bytes(reg[8:16], "big")
        except OSError:
            pass
    return 0


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

    def describe(self) -> dict:
        total = int(Path("/proc/meminfo").read_text().split()[1]) * 1024
        return {"models": {name: model.describe() for name, model in self.models.items()},
                "memory": {"available_bytes": memory_available(), "total_bytes": total,
                           "mla_total_bytes": self._mla_total, "mla_loaded_bytes": self._loaded_bytes()}}

    def _loaded_bytes(self) -> int:
        """Graph bytes this app has on the MLA. Other applications' models are not visible here."""
        return sum(model.graph_bytes(s) for model in self.models.values() for s in model.loaded_seq_lens)

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
            if memory_available() < model.fixed_bytes() + self.HEADROOM:
                raise Refused(
                    f"not enough board memory for {name}'s embedding table "
                    f"({memory_available() >> 20} MB available).")
            # One model at a time: whatever is loaded goes first, this model included.
            others = [other for other in self.models.values() if other.process is not None]
            progress = Progress(model.loading_stages(wanted, [other.name for other in others]))
            model.state, model.progress = "loading", progress
            process = LayaProcess(model._laya, str(model.path), ",".join(map(str, wanted)))
            first_graph = len(others) + 1           # stages: unloads, reading, graphs, warm-up
            elfs = [model.config["elfs"][str(s)] for s in wanted]

            def follow(line: str):
                """The runtime says when it starts and finishes each graph."""
                for position, elf in enumerate(elfs):
                    if line.endswith(elf) and line.startswith("Loading model"):
                        progress.enter(first_graph + position)
                    elif line.endswith(elf) and line.startswith("Done loading"):
                        progress.enter(first_graph + position + 1)

            try:
                for position, other in enumerate(others):
                    progress.enter(position)
                    self._unload(other, show=other is not model)
                model.state, model.progress = "loading", progress
                progress.enter(len(others))
                process.start(follow)
            except (RuntimeDied, OSError):
                model.state, model.progress = "unloaded", None
                raise
            model.process, model.loaded_seq_lens = process, wanted
            model.state, model.progress = "loaded", None

    def _unload(self, model: Model, show: bool = True):
        if model.process is not None:
            if show:
                model.state, model.progress = "unloading", Progress([("Unloading", SECONDS_TO_UNLOAD)])
            model.process.stop()   # waits for a request in flight
        model.process, model.loaded_seq_lens = None, []
        model.state, model.progress = "unloaded", None

    def unload(self, name: str):
        with self._admin:
            self._unload(self.models[name])

    def request(self, name: str, payload: dict) -> dict:
        model = self.models[name]
        process = model.process
        if model.state != "loaded" or process is None:
            raise NotLoaded(f"the {name} model is not loaded")
        if not process.running:
            self.unload(name)
            raise RuntimeDied(f"the {name} model's runtime has stopped; load it again")
        return process.request(payload)

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
            names = ["laya_config.json", *model.config["elfs"].values(),
                     *(model.config[key] for key in ("token_embeddings", "act_tail", "tokenizer"))]
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

    def _url(self, path: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{path}"

    def directory(self, name: str) -> Path:
        return self.root / ("model" if name == "general" else f"model-{name}")

    def _refresh(self):
        try:
            request = urllib.request.Request(self._url("models.json"), headers={"User-Agent": "sima-laya"})
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
        request = urllib.request.Request(self._url(path), headers={"User-Agent": "sima-laya"})
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


PAGES = {"/": "debate.html", "/debate": "debate.html", "/questions": "index.html", "/games": "games.html",
         "/games/blackjack": "blackjack.html", "/games/snake": "snake.html", "/games/sudoku": "sudoku.html",
         "/models": "models.html"}
# Scripts, the shared stylesheet, and the brand images and fonts it uses.
ASSET = re.compile(r"/static/((?:brand/|fonts/)?[a-z][a-z0-9-]*\.(js|css|png|svg|woff2))")
ASSET_TYPES = {"js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8", "png": "image/png",
               "svg": "image/svg+xml", "woff2": "font/woff2"}


def make_handler(manager: ModelManager, hub: Hub | None):
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
            elif path == "/api/hub":
                self._json(200, hub.describe() if hub else {"repo": None, "listed": True, "models": {}, "error": None})
            elif (asset := ASSET.fullmatch(path)) and (STATIC / asset[1]).is_file():
                # Images and fonts do not change between deploys; code does.
                self._send(200, (STATIC / asset[1]).read_bytes(), ASSET_TYPES[asset[2]],
                           cache="max-age=86400" if asset[2] in ("png", "svg", "woff2") else "no-store")
            else:
                self._json(404, {"error": "not found"})

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
            name = request.get("model", "general")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/media/nvme/laya/model", help="compiled model directory")
    ap.add_argument("--laya", default="/media/nvme/laya/laya", help="the laya runtime binary")
    ap.add_argument("--seq-lens", help="offer only these compiled sequence lengths, e.g. 128,512")
    ap.add_argument("--game-model", help="compiled model directory for the games page")
    ap.add_argument("--extra-model", action="append", default=[], metavar="NAME=DIR",
                    help="another question model (repeatable)")
    ap.add_argument("--preload", default="general", metavar="NAME",
                    help="the model to load at startup, or 'none' (default: %(default)s). "
                         "Another one is loaded, in its place, from the pages.")
    ap.add_argument("--hub", default="TDoSiMa/sima-laya", metavar="REPO",
                    help="Hugging Face repository the Models page offers downloads from, or 'none' "
                         "(default: %(default)s)")
    ap.add_argument("--root", help="where downloaded models go: `model` and `model-<name>` in this "
                                   "directory (default: the directory holding --model)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8095)
    args = ap.parse_args()

    # A board may have no model yet: the Models page can then fetch one.
    sources = {}
    if (Path(args.model) / "laya_config.json").is_file():
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

    preload = [] if args.preload == "none" else [args.preload]
    if preload == ["general"] and "general" not in manager.models:
        print("laya webapp: the general model is not on this board; starting with nothing loaded", flush=True)
        preload = []
    for name in preload:
        if name not in manager.models:
            sys.exit(f"laya webapp: --preload names {name!r}, which is not one of {list(sources)}")
        try:
            manager.load(name)
        except (RuntimeDied, Refused, OSError) as error:
            manager.stop()
            sys.exit(f"laya webapp: cannot load the {name} model: {error}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager, hub))

    def on_sigterm(*_):
        # Leave through the same path as Ctrl-C, so the runtimes release their models on the MLA.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    print(f"Laya playground on http://{args.host}:{args.port}  models: {', '.join(sources) or 'none'}; "
          f"loaded: {', '.join(preload) or 'none'}; hub: {args.hub}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.stop()


if __name__ == "__main__":
    main()
