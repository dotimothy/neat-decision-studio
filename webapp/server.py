#!/usr/bin/env python3
"""Laya decision playground: a web front end for the Laya runtime on a Modalix DevKit.

Runs on the board with nothing but the Python standard library. It keeps a `laya serve`
process alive for the loaded model, so its compiled graphs stay on the MLA, and forwards each
browser request to it as one JSON line.

    /           Debate, the default: a yes-or-no question answered live as it is typed
    /questions  question answering with your own questions and options
    /games      games a model plays: Dino Arena, Blackjack, Snake
    /models     the model manager: what is on the MLA, with load and unload

One model is on the MLA at a time. Models are loaded only on request (the Models page, or the
Load button a page shows when the model it wants is not the loaded one), and loading one
unloads whichever was there; `--preload` chooses what is loaded at startup.

    python3 webapp/server.py --model /media/nvme/laya/model --laya /media/nvme/laya/laya
"""
import argparse
import json
import re
import signal
import socket
import subprocess
import sys
import threading
import time
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

    def stop(self):
        for model in self.models.values():
            self._unload(model)


PAGES = {"/": "debate.html", "/debate": "debate.html", "/questions": "index.html", "/games": "games.html",
         "/games/blackjack": "blackjack.html", "/games/snake": "snake.html", "/games/sudoku": "sudoku.html",
         "/models": "models.html"}
# Scripts, the shared stylesheet, and the brand images and fonts it uses.
ASSET = re.compile(r"/static/((?:brand/|fonts/)?[a-z][a-z0-9-]*\.(js|css|png|svg|woff2))")
ASSET_TYPES = {"js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8", "png": "image/png",
               "svg": "image/svg+xml", "woff2": "font/woff2"}


def make_handler(manager: ModelManager):
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
            elif (asset := ASSET.fullmatch(path)) and (STATIC / asset[1]).is_file():
                # Images and fonts do not change between deploys; code does.
                self._send(200, (STATIC / asset[1]).read_bytes(), ASSET_TYPES[asset[2]],
                           cache="max-age=86400" if asset[2] in ("png", "svg", "woff2") else "no-store")
            else:
                self._json(404, {"error": "not found"})

        def _body(self) -> dict | None:
            """The request's JSON object, or None after an error response has been sent."""
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
            if name not in manager.models:
                self._json(404, {"error": f"there is no model named {name!r}"})
                return None
            request["model"] = name
            return request

        def do_POST(self):
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
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8095)
    args = ap.parse_args()

    sources = {"general": (args.model, args.seq_lens)}
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

    preload = [] if args.preload == "none" else [args.preload]
    for name in preload:
        if name not in manager.models:
            sys.exit(f"laya webapp: --preload names {name!r}, which is not one of {list(sources)}")
        try:
            manager.load(name)
        except (RuntimeDied, Refused, OSError) as error:
            manager.stop()
            sys.exit(f"laya webapp: cannot load the {name} model: {error}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manager))

    def on_sigterm(*_):
        # Leave through the same path as Ctrl-C, so the runtimes release their models on the MLA.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    print(f"Laya playground on http://{args.host}:{args.port}  models: {', '.join(sources)}; "
          f"loaded: {', '.join(preload) or 'none'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.stop()


if __name__ == "__main__":
    main()
