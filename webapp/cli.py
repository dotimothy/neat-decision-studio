#!/usr/bin/env python3
"""Neat Decision Studio in a terminal (`./run.sh --cli`).

Talks to the running app's HTTP API, the same one the pages use, so models are downloaded,
loaded, unloaded and deleted here exactly as on the Models page, and a question is one line.
Standard library only.

    python3 webapp/cli.py --url http://localhost:8095              the prompt
    python3 webapp/cli.py --url ... --ask "Is the sky blue?"       one answer, then exit
    echo "Is the sky blue?" | python3 webapp/cli.py --url ...      one answer per line

A line that is not a command is a yes-or-no question, answered as on the Debate page: the
model decides again at every keystroke, and its answer is drawn under the line while it is
typed. With a state set (`/state ...`), a line is a yes-or-no question about that state, and
`/choice` and `/score` ask the other two kinds.
"""
import argparse
import codecs
import json
import os
import select
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

HISTORY = os.path.expanduser("~/.neat_decision_history")
try:
    import readline
except ImportError:                      # the prompt still works, without recall
    readline = None

try:
    import termios
    import tty
except ImportError:                      # no terminal control: the prompt answers on Enter only
    termios = None

# Colour, as run.sh and setup.sh use it; plain when not a terminal, or NO_COLOR is set.
if sys.stdout.isatty() and os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb":
    RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
    TEAL, GREEN, LIME, BLUE, ORANGE, INK = ("\033[38;2;61;179;138m", "\033[38;2;74;168;54m", "\033[38;2;154;190;30m",
                                            "\033[38;2;58;125;216m", "\033[38;2;223;108;30m", "\033[38;2;60;66;74m")
    ACCENT, MUTED, OK, WARN, ERR = "\033[38;2;47;212;192m", "\033[38;2;140;150;160m", "\033[38;2;53;196;137m", \
                                   "\033[38;2;224;173;74m", "\033[38;2;239;91;98m"
else:
    RESET = BOLD = DIM = TEAL = GREEN = LIME = BLUE = ORANGE = INK = ACCENT = MUTED = OK = WARN = ERR = ""


def rl(code: str) -> str:
    """A colour code inside an input() prompt: readline must not count it as width."""
    return f"\001{code}\002" if code and readline else code


def banner(url: str):
    print(f"""
        {TEAL}▲{RESET}
       {GREEN}███{RESET}
      {GREEN}█████{RESET}
   {BLUE}◀████{INK}█{LIME}████▶{RESET}
      {ORANGE}█████{RESET}
       {ORANGE}███{RESET}
        {ORANGE}▼{RESET}

   {BOLD}NEAT{RESET} {ACCENT}{BOLD}Decision Studio{RESET}  {MUTED}· terminal{RESET}
   {MUTED}System-1 decisions on the Modalix MLA · {url}{RESET}
""")


class Refused(RuntimeError):
    pass


class Client:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def call(self, path: str, payload: dict | None = None, timeout: float = 600) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:        # the app answers errors as JSON too
            try:
                body = json.loads(error.read())
            except ValueError:
                raise Refused(f"the app answered {error.code}") from error
            if path == "/api/predict" and body.get("not_loaded"):
                raise Refused("no model is on the MLA; /load one")
            raise Refused(body.get("error", f"the app answered {error.code}")) from error
        except OSError as error:
            raise Refused(f"the app at {self.url} did not answer: {error}") from error

    def info(self) -> dict:
        return self.call("/api/info", timeout=10)

    def hub(self) -> dict:
        return self.call("/api/hub", timeout=10)

    # The model questions go to when several are loaded (/use), for as long as it stays loaded.
    chosen: str | None = None

    def loaded_all(self) -> list[str]:
        """Every model on the MLA, oldest first."""
        info = self.info()
        return info.get("loaded") or [name for name, model in info["models"].items() if model["loaded"]]

    def loaded(self) -> str | None:
        """The model that answers: the one chosen with /use, else the first that was loaded."""
        names = self.loaded_all()
        return self.chosen if self.chosen in names else next(iter(names), None)


def size(count: int) -> str:
    return f"{count / (1 << 30):.2f} GB" if count >= 1 << 30 else f"{round(count / (1 << 20))} MB"


def bar(fraction: float, width: int = 28, colour: str = ACCENT) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return f"{colour}{'█' * filled}{RESET}{DIM}{'░' * (width - filled)}{RESET}"


def progress_line(fraction: float, text: str):
    columns = os.get_terminal_size().columns if sys.stdout.isatty() else 100
    line = f"  {bar(fraction, 24)} {round(fraction * 100):3d}%  {MUTED}{text}{RESET}"
    sys.stdout.write("\r\033[K" + line[: columns + 60])
    sys.stdout.flush()


# ----------------------------------------------------------------------------- models

def catalog(client: Client) -> dict:
    """Every model once: what the board says about it and what Hugging Face lists."""
    info, hub = client.info(), client.hub()
    for _ in range(40):                      # the app fetches the list in the background
        if hub.get("listed") or hub.get("error") or not hub.get("repo"):
            break
        time.sleep(0.25)
        hub = client.hub()
    names = list(info["models"]) + [name for name in hub["models"] if name not in info["models"]]
    return {name: (info["models"].get(name), hub["models"].get(name)) for name in names}


def title(name: str, local: dict | None, remote: dict | None) -> str:
    return (remote or {}).get("title") or (local or {}).get("checkpoint") or name


def show_models(client: Client):
    models = catalog(client)
    if not models:
        print(f"{MUTED}  No model on this board, and no list from Hugging Face.{RESET}")
        return
    print()
    for name, (local, remote) in models.items():
        if local and local["loaded"]:
            where = f"{OK}loaded on the MLA{RESET}"
        elif local:
            where = f"{ACCENT}on this board{RESET}"
        elif remote["state"] == "downloading":
            where = f"{WARN}downloading{RESET}"
        else:
            where = f"{MUTED}on Hugging Face{RESET}"
        speed = {graph["seq_len"]: graph.get("latency_ms") for graph in (remote or {}).get("graphs", [])}
        graphs = local["graphs"] if local else remote["graphs"]
        bits = ", ".join(f"{g['seq_len']} tokens" + (f" {speed[g['seq_len']]:g} ms" if speed.get(g["seq_len"]) else "") for g in graphs)
        total = sum(g["bytes"] for g in local["graphs"]) + local["fixed_bytes"] if local else remote["bytes"]
        print(f"  {BOLD}{name:16s}{RESET} {title(name, local, remote):22s} {where}")
        print(f"  {'':16s} {MUTED}{size(total)} · {bits}{RESET}")
        if remote and remote.get("about"):
            print(f"  {'':16s} {MUTED}{remote['about']}{RESET}")
    print(f"\n{MUTED}  /load NAME · /download NAME · /delete NAME · /unload{RESET}\n")


def follow(client: Client, path: str, payload: dict, watch) -> dict:
    """Post an action and draw its progress, which `watch()` reads from the app, until it returns."""
    result = {}

    def post():
        try:
            result["body"] = client.call(path, payload)
        except Refused as error:
            result["error"] = str(error)

    thread = threading.Thread(target=post, daemon=True)
    thread.start()
    while thread.is_alive():
        try:
            state = watch()
        except Refused:
            state = None
        if state and sys.stdout.isatty():
            progress_line(*state)
        thread.join(0.2)
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K")
    if "error" in result:
        raise Refused(result["error"])
    return result["body"]


def load(client: Client, name: str, seq_lens: list[int] | None = None):
    def watch():
        progress = client.info()["models"].get(name, {}).get("progress")
        return (progress["fraction"], f"{progress['stage']} · step {progress['step']} of {progress['steps']}") if progress else None

    started = time.monotonic()
    body = follow(client, "/api/models/load", {"model": name, **({"seq_lens": seq_lens} if seq_lens else {})}, watch)
    graphs = ", ".join(map(str, body["models"][name]["loaded_seq_lens"]))
    print(f"{OK}✔{RESET} {name} is on the MLA ({graphs}-token graphs) {MUTED}in {time.monotonic() - started:.1f} s{RESET}")


def download(client: Client, name: str):
    """Fetch a model from Hugging Face onto the board; Ctrl-C cancels and keeps what arrived."""
    client.call("/api/hub/download", {"model": name})
    try:
        while True:
            model = client.hub()["models"].get(name, {})
            if model.get("state") != "downloading":
                break
            progress = model.get("progress")
            if progress and sys.stdout.isatty():
                rate = f" · {progress['bytes_per_second'] / 1e6:.1f} MB/s" if progress["bytes_per_second"] else ""
                left = f" · about {progress['seconds_left']} s left" if progress["seconds_left"] is not None else ""
                progress_line(progress["fraction"], f"{size(progress['done_bytes'])} of {size(progress['total_bytes'])}{rate}{left}")
            time.sleep(0.4)
    except KeyboardInterrupt:
        client.call("/api/hub/cancel", {"model": name})
        sys.stdout.write("\r\033[K")
        print(f"{WARN}⚠{RESET} cancelled; what had arrived is kept, and /download {name} continues from there")
        return False
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[K")
    if model.get("state") == "on_board":
        print(f"{OK}✔{RESET} {name} is on this board")
        return True
    print(f"{ERR}✘{RESET} {model.get('error') or 'the download stopped'}")
    return False


def menu(items: list[tuple[str, str]], prompt: str) -> str | None:
    """One of `items` (label, value) by number; None when cancelled."""
    print(f"{MUTED}{prompt}{RESET}")
    for index, (label, _) in enumerate(items, 1):
        print(f"  {DIM}{index}.{RESET} {label}")
    try:
        raw = input(f"{MUTED}  number (blank to cancel) ▸ {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return items[int(raw) - 1][1] if raw.isdigit() and 1 <= int(raw) <= len(items) else None


def choose_model(client: Client) -> str | None:
    """Pick a model to put on the MLA, downloading it first if it is only on Hugging Face."""
    models = catalog(client)
    if not models:
        print(f"{WARN}⚠{RESET} no model on this board, and no list from Hugging Face")
        return None
    items = []
    for name, (local, remote) in models.items():
        total = sum(g["bytes"] for g in local["graphs"]) + local["fixed_bytes"] if local else remote["bytes"]
        where = "on this board" if local else "on Hugging Face, download first"
        items.append((f"{BOLD}{title(name, local, remote)}{RESET}  {MUTED}{name} · {size(total)} · {where}{RESET}", name))
    name = menu(items, "Which model?")
    if name is None:
        return None
    if models[name][0] is None and not download(client, name):
        return None
    load(client, name)
    return name


# ----------------------------------------------------------------------------- questions

YES_NO = {"type": "choice", "instructions": "Is the answer yes or no?", "criteria": ["yes", "no"]}


def parse_question(kind: str, rest: str) -> dict:
    """`/choice Which team? | billing: invoices | platform: outages`, `/score How urgent? | low | high`."""
    parts = [part.strip() for part in rest.split("|")]
    instructions, options = parts[0], [part for part in parts[1:] if part]
    if not instructions:
        raise Refused(f"/{kind} needs a question" + ("" if kind == "yesno" else " and options: /" + kind + " QUESTION | A | B | C"))
    if kind == "yesno":
        return {"type": "noul", "instructions": instructions}
    if len(options) < 2:
        raise Refused(f"/{kind} needs at least two options: /{kind} QUESTION | A | B | C")
    if kind == "choice" and all(":" in option for option in options):      # name: what it covers
        return {"type": "choice", "instructions": instructions,
                "criteria": {name.strip(): about.strip() for name, about in (option.split(":", 1) for option in options)}}
    return {"type": kind, "instructions": instructions, "criteria": options}


def show_answer(answer: dict, usage: dict, trip_ms: float):
    kind = answer["type"]
    if kind == "noul":
        rows, top = [("yes", answer["noul"]), ("no", 1 - answer["noul"])], "yes" if answer["noul"] >= 0.5 else "no"
    elif kind == "score":
        rows = [(f"{level} · {answer['legend'][level]}", p) for level, p in answer["probabilities"].items()]
        top = max(answer["probabilities"], key=answer["probabilities"].get)
        top = f"{top} · {answer['legend'][top]}"
    else:
        rows, top = list(answer["probabilities"].items()), answer["choice"]
    width = max(len(label) for label, _ in rows)
    print()
    for label, p in rows:
        mark = BOLD if label == top else ""
        print(f"  {mark}{label:{width}s}{RESET}  {bar(p, colour=ACCENT if label == top else MUTED)}  {mark}{p * 100:5.1f}%{RESET}")
    verdict = f"{answer['score']:.2f} of {len(rows) - 1}" if kind == "score" else top
    print(f"\n  {ACCENT}{BOLD}{verdict}{RESET}  {MUTED}{usage['mla_ms']:.1f} ms on the MLA · {usage['tokens']} tokens on the "
          f"{usage['seq_len']}-token graph · {trip_ms:.0f} ms round trip{RESET}\n")


def compare(session: "Session", text: str):
    """One yes-or-no question to every loaded model at once, a line a model."""
    payload = ({"yes_no": text} if session.state is None
               else {"state": session.state, "questions": {"answer": {"type": "noul", "instructions": text}}})
    sent = time.perf_counter()
    body = session.client.call("/api/compare", payload)
    trip = (time.perf_counter() - sent) * 1e3
    if session.raw:
        print(json.dumps(body, indent=1))
        return
    width = max(len(name) for name in body["models"])
    verdicts = []
    print()
    for name in body["models"]:
        result = body["results"][name]
        if "error" in result:
            print(f"  {name:{width}s}  {ERR}{result['error']}{RESET}")
            continue
        answer = result["answers"]["answer"]
        yes = result.get("yes", answer.get("noul"))
        verdicts.append("yes" if yes >= 0.5 else "no")
        print(f"  {BOLD}{name:{width}s}{RESET}  {bar(yes, colour=ACCENT if yes >= 0.5 else MUTED)}  {BOLD}{verdicts[-1]:3s}{RESET} {yes * 100:5.1f}% yes"
              f"  {MUTED}{result['usage']['mla_ms']:.1f} ms on the MLA{RESET}")
    agree = len(set(verdicts)) == 1 and len(verdicts) > 1
    print(f"\n  {ACCENT if agree else WARN}{BOLD}{'All agree: ' + verdicts[0] if agree else 'The models differ' if len(verdicts) > 1 else 'One model answered'}{RESET}"
          f"  {MUTED}{trip:.0f} ms round trip{RESET}\n")


class Session:
    def __init__(self, client: Client, budget: int = 0, raw: bool = False):
        self.client, self.budget, self.raw = client, budget, raw
        self.state: str | None = None        # the text questions are about; None is Debate
        self.last: dict | None = None        # the last request, for /bench
        self._kinds: dict = {}               # model name -> "laya" or "clm"

    def ask(self, state, question: dict) -> dict:
        model = self.client.loaded()
        if model is None:
            raise Refused("no model is on the MLA; /load one")
        payload = {"model": model, "state": state, "questions": {"q": question}}
        if self.budget:
            payload["max_len"] = self.budget
        self.last = payload
        sent = time.perf_counter()
        body = self.client.call("/api/predict", payload)
        trip = (time.perf_counter() - sent) * 1e3
        if self.raw:
            print(json.dumps(body, indent=1))
        else:
            show_answer(body["answers"]["q"], body["usage"], trip)
        return body

    def line(self, text: str) -> dict:
        """A plain line: a yes-or-no question, about the state if there is one."""
        return self.ask(*self.plain(text))

    def plain(self, text: str, model: str | None = None) -> tuple:
        """The state and question a plain line stands for.

        With no state, Laya is asked as the Debate page asks it; CLM is asked the way its heads
        were trained, the question itself as a yes-or-no question with nothing before it.
        """
        if self.state is None:
            if self.kind(model) == "clm":
                return "", {"type": "noul", "instructions": text}
            return "Question: " + text, YES_NO
        return self.state, {"type": "noul", "instructions": text}

    def kind(self, model: str | None) -> str:
        """Which kind of model `model` (default: the loaded one) is; asked once per model."""
        model = model or self.client.loaded()
        if model not in self._kinds:
            self._kinds[model] = (self.client.info()["models"].get(model) or {}).get("kind", "laya")
        return self._kinds[model]

    def peek(self, text: str, model: str) -> dict:
        """The same decision without printing it: what the live prompt draws while a line is typed."""
        state, question = self.plain(text, model)
        payload = {"model": model, "state": state, "questions": {"q": question}}
        if self.budget:
            payload["max_len"] = self.budget
        sent = time.perf_counter()
        body = self.client.call("/api/predict", payload)
        self.last = payload
        answer = body["answers"]["q"]
        yes = answer["noul"] if answer["type"] == "noul" else answer["probabilities"]["yes"]
        return {"yes": yes, "usage": body["usage"], "trip": (time.perf_counter() - sent) * 1e3, "body": body}


class LivePrompt:
    """A prompt whose line is answered while it is typed.

    Every keystroke asks the model again, as the Debate page does: the newest text wins, and
    the answer is drawn in the three lines under the prompt. Enter keeps the answer in the
    scrollback. It needs a terminal it can put into character mode; without one the plain
    prompt is used, which answers on Enter.
    """

    PANEL = 3

    def __init__(self, session: Session):
        self.session, self.model = session, None
        self.text, self.label, self.width = "", "", 0
        self.result = None                   # (the text it is for, a peek() result or an error message)
        self.decisions = 0
        self.history = []
        try:
            for line in open(HISTORY, encoding="utf-8", errors="replace").read().splitlines():
                if line and line != "_HiStOrY_V2_":
                    self.history.append(line.replace("\\040", " "))
        except OSError:
            pass
        self.lock = threading.Lock()         # the terminal: typing and answers both draw on it
        self.wake = threading.Condition()
        self.want, self.active = None, False
        self.keys, self.decoder = [], codecs.getincrementaldecoder("utf-8")("replace")     # typed, not yet handled
        threading.Thread(target=self._work, daemon=True).start()

    @staticmethod
    def usable() -> bool:
        return termios is not None and sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"

    def _work(self):
        """Ask about the newest text; whatever was typed while a decision was being made replaces it."""
        while True:
            with self.wake:
                while self.want is None:
                    self.wake.wait()
                text, self.want = self.want, None
            try:
                result = self.session.peek(text, self.model) if self.model else "no model is on the MLA; /load one"
                self.decisions += 1
            except Refused as error:
                result = str(error)
            except (KeyError, TypeError):
                result = "the app's answer was not understood"
            with self.lock:
                self.result = (text, result)
                if self.active:
                    self._paint()

    def _panel(self, columns: int) -> list:
        text = self.text.strip()
        if not text:
            return [f"{MUTED}  Type a yes-or-no question: Laya decides at every keystroke.{RESET}"[: columns + 20], "", ""]
        if text.startswith("/"):
            return [f"{MUTED}  A command: Enter runs it, /help lists them.{RESET}", "", ""]
        if self.result is None or isinstance(self.result[1], str):
            return [f"{MUTED}  {self.result[1] if self.result else 'deciding…'}{RESET}"[: columns + 20], "", ""]
        result, room = self.result[1], max(6, min(28, columns - 22))
        yes, usage = result["yes"], result["usage"]
        rows = []
        for label, p in (("yes", yes), ("no", 1 - yes)):
            top = p >= 0.5 if label == "yes" else p > 0.5
            rows.append(f"  {BOLD if top else ''}{label:3s}{RESET}  {bar(p, room, ACCENT if top else MUTED)}  {BOLD if top else ''}{p * 100:5.1f}%{RESET}")
        note = f"  {MUTED}{usage['mla_ms']:.1f} ms on the MLA · {usage['tokens']} tokens · {self.decisions} decisions{RESET}"
        return rows + [note if columns >= 50 else f"  {MUTED}{usage['mla_ms']:.1f} ms{RESET}"]

    def _paint(self):
        """The prompt line and the panel under it, leaving the cursor at the end of the line."""
        columns = shutil.get_terminal_size().columns
        shown = self.text[-max(8, columns - self.width - 2):]        # a long line scrolls sideways
        out = "\r\033[K" + self.label + shown
        for line in self._panel(columns):
            out += "\n\033[K" + line
        out += f"\033[{self.PANEL}A\r\033[{self.width + len(shown)}C"
        sys.stdout.write(out)
        sys.stdout.flush()

    def _changed(self):
        with self.lock:
            self._paint()
        text = self.text.strip()
        if text and not text.startswith("/"):
            with self.wake:
                self.want = text
                self.wake.notify()

    def _key(self, fd: int, wait: float | None = None):
        """The next character typed, read from the terminal itself so that the keys of one
        sequence (an arrow is three) can be told from keys typed one after another. None when
        nothing arrives within `wait` seconds, "" at the end of input."""
        while not self.keys:
            if wait is not None and not select.select([fd], [], [], wait)[0]:
                return None
            data = os.read(fd, 1024)
            if not data:
                return ""
            self.keys.extend(self.decoder.decode(data))
        return self.keys.pop(0)

    def read(self, label: str, width: int, model: str | None):
        """One line. Returns it, None at the end of input, and "" after Ctrl-C."""
        self.label, self.width, self.model = label, width, model
        self.text, self.result, recall = "", None, len(self.history)
        fd = sys.stdin.fileno()
        before = termios.tcgetattr(fd)
        outcome = ""
        try:
            tty.setcbreak(fd)
            with self.lock:
                self.active = True
                self._paint()
            while True:
                key = self._key(fd)
                if key == "" or (key == "\x04" and not self.text):
                    outcome = None
                    break
                if key in "\r\n":
                    outcome = self.text.strip()
                    break
                if key == "\x1b":                                  # an arrow, or some other key's sequence
                    tail = ""
                    while len(tail) < 8:
                        more = self._key(fd, 0.03)
                        if not more:
                            break
                        tail += more
                        if more.isalpha() or more == "~":
                            break
                    if tail in ("[A", "[B") and self.history:
                        recall = max(0, min(len(self.history), recall + (-1 if tail == "[A" else 1)))
                        self.text = self.history[recall] if recall < len(self.history) else ""
                    else:
                        continue
                elif key in "\x7f\b":
                    self.text = self.text[:-1]
                elif key == "\x15":                                 # Ctrl-U: the whole line
                    self.text = ""
                elif key == "\x17":                                 # Ctrl-W: the last word
                    self.text = self.text.rstrip().rpartition(" ")[0]
                    self.text += " " if self.text else ""
                elif key.isprintable():
                    self.text += key
                else:
                    continue
                self._changed()
        except KeyboardInterrupt:
            outcome = ""
        finally:
            with self.lock:
                self.active = False
                # The line stays; the panel under it is cleared, and the cursor goes to the line after.
                sys.stdout.write("\r\033[K" + self.label + self.text + "\n\033[K" * self.PANEL + f"\033[{self.PANEL}A\r\n")
                sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, before)
        if outcome:
            self.history = [line for line in self.history if line != outcome] + [outcome]
        return outcome

    def answered(self, text: str):
        """The live answer for `text`, if the last one drawn was for exactly that."""
        if self.result and self.result[0] == text and not isinstance(self.result[1], str):
            return self.result[1]
        return None

    def save(self):
        try:
            with open(HISTORY, "w", encoding="utf-8") as out:
                out.write("\n".join(self.history[-1000:]) + "\n")
        except OSError:
            pass

    def bench(self, runs: int):
        if self.last is None:
            raise Refused("ask something first; /bench repeats the last question")
        mla, trips = [], []
        for index in range(runs):
            sent = time.perf_counter()
            body = self.client.call("/api/predict", self.last)
            trips.append((time.perf_counter() - sent) * 1e3)
            mla.append(body["usage"]["mla_ms"])
            if sys.stdout.isatty():
                progress_line((index + 1) / runs, f"{index + 1} of {runs}")
        if sys.stdout.isatty():
            sys.stdout.write("\r\033[K")
        mla.sort(); trips.sort()
        p95 = lambda values: values[min(len(values) - 1, int(len(values) * 0.95))]
        print(f"\n  {BOLD}{runs} decisions{RESET} {MUTED}on the {body['usage']['seq_len']}-token graph, {body['usage']['tokens']} tokens{RESET}")
        print(f"  on the MLA    mean {ACCENT}{BOLD}{sum(mla) / runs:.2f} ms{RESET}  min {mla[0]:.2f}  p95 {p95(mla):.2f}  max {mla[-1]:.2f}")
        print(f"  round trip    mean {sum(trips) / runs:.2f} ms  min {trips[0]:.2f}  p95 {p95(trips):.2f}  max {trips[-1]:.2f}"
              f"  {MUTED}({1000 * runs / sum(trips):.0f} decisions a second){RESET}\n")


HELP = f"""{MUTED}
  A line that is not a command is a yes-or-no question. The model on the MLA decides again
  at every keystroke, and its answer is drawn under the line while you type.

  /models                     every model, on this board or on Hugging Face
  /load [NAME] [128,512]      put a model on the MLA, beside any already there (no name: choose from a list)
  /unload [NAME|all]          take a model off the MLA (no name: the one that answers)
  /use NAME                   with several loaded: the one that answers from now on
  /compare QUESTION           ask every loaded model at once and show them side by side
  /download NAME              fetch a model from Hugging Face onto this board (Ctrl-C cancels)
  /delete NAME                remove a model's files from this board
  /reset                      reset the accelerator: every model comes off the MLA, other apps' too

  /state TEXT                 the text the next questions are about (/state alone: back to plain questions)
  /yesno QUESTION             a yes-or-no question about the state
  /choice QUESTION | A | B    one of several options; "name: what it covers" describes an option
  /score QUESTION | LOW | HIGH  a level on a scale
  /budget [N]                 how many tokens a question, its options and the state may take
  /bench [N]                  repeat the last question N times (default 50) and time it
  /json                       show answers as the raw JSON the API returns, or stop doing so

  /help                       this help
  /quit                       leave (also Ctrl-D){RESET}
"""


def command(session: Session, line: str) -> bool:
    """Run a slash command. False ends the session."""
    client = session.client
    name, _, rest = line[1:].partition(" ")
    name, rest = name.lower(), rest.strip()
    if name in ("quit", "exit", "q", "bye"):
        return False
    if name in ("help", "h", "?"):
        print(HELP)
    elif name in ("models", "ls"):
        show_models(client)
    elif name == "load":
        if not rest:
            choose_model(client)
        else:
            target, _, sizes = rest.partition(" ")
            load(client, target, [int(part) for part in sizes.replace(",", " ").split()] or None)
    elif name == "unload":
        names = client.loaded_all()
        if not names:
            raise Refused("no model is on the MLA")
        if rest == "all":
            client.call("/api/models/unload", {"model": names[0], "all": True})
            print(f"{OK}✔{RESET} {', '.join(names)} {'is' if len(names) == 1 else 'are'} off the MLA")
        else:
            model = rest or client.loaded()
            if model not in names:
                raise Refused(f"{model} is not on the MLA; loaded: {', '.join(names)}")
            client.call("/api/models/unload", {"model": model})
            print(f"{OK}✔{RESET} {model} is off the MLA")
    elif name == "use":
        names = client.loaded_all()
        if rest not in names:
            raise Refused(f"/use needs a loaded model; loaded: {', '.join(names) or 'none'}")
        client.chosen = rest
        print(f"{OK}✔{RESET} {rest} answers from now on")
    elif name in ("compare", "all"):
        if not rest:
            raise Refused("/compare needs a question")
        compare(session, rest)
    elif name in ("download", "hub"):
        if not rest:
            raise Refused("/download needs a model name; /models lists them")
        download(client, rest)
    elif name in ("delete", "rm", "remove"):
        if not rest:
            raise Refused("/delete needs a model name; /models lists them")
        try:
            sure = input(f"{WARN}  Delete {rest} from this board's disk? [y/N] {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sure = ""
            print()
        if sure in ("y", "yes"):
            client.call("/api/models/delete", {"model": rest})
            print(f"{OK}✔{RESET} {rest} is deleted from this board")
    elif name in ("reset", "reset-mla"):
        print(f"{WARN}  This restarts the board's MLA services. Every model comes off the accelerator: this app's and any\n"
              f"  other application's, such as Neat GenAI Studio, which then has to load its models again.{RESET}")
        try:
            sure = input(f"{WARN}  Reset the accelerator? [y/N] {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sure = ""
            print()
        if sure in ("y", "yes"):
            print(f"{MUTED}  Resetting…{RESET}", flush=True)
            client.call("/api/mla/reset", {"token": rest} if rest else {})
            print(f"{OK}✔{RESET} The accelerator is reset; /load a model")
    elif name == "state":
        session.state = rest or None
        print(f"{MUTED}  {'Questions are now about that text.' if rest else 'Back to plain yes-or-no questions.'}{RESET}")
    elif name in ("yesno", "choice", "score"):
        if session.state is None:
            raise Refused(f"/{name} asks about a state; set one first with /state TEXT")
        session.ask(session.state, parse_question(name, rest))
    elif name in ("budget", "tokens"):
        if rest:
            session.budget = max(0, int(rest))
        print(f"{MUTED}  token budget: {session.budget or 'the largest loaded graph'}{RESET}")
    elif name in ("bench", "benchmark"):
        session.bench(int(rest) if rest else 50)
    elif name == "json":
        session.raw = not session.raw
        print(f"{MUTED}  raw JSON {'on' if session.raw else 'off'}{RESET}")
    else:
        raise Refused(f"there is no /{name}; /help lists the commands")
    return True


def repl(session: Session):
    client = session.client
    live = LivePrompt(session) if LivePrompt.usable() else None
    if readline and not live:
        try:
            readline.read_history_file(HISTORY)
        except OSError:
            pass
        readline.set_history_length(1000)
    print(f"{MUTED}  /help for commands, /quit to leave.{RESET}\n")
    while True:
        try:
            loaded = client.loaded()
        except Refused as error:
            print(f"{ERR}✘{RESET} {error}")
            break
        name = loaded or "no model"
        others = len(client.loaded_all()) - 1 if loaded else 0
        if others > 0:
            name += f" +{others}"
        if live:
            about = f" {ACCENT}· state{RESET}" if session.state else ""
            line = live.read(f"{ACCENT}{BOLD}laya{RESET} {DIM}[{name}]{RESET}{about} ▸ ",
                             len(f"laya [{name}]{' · state' if session.state else ''} ▸ "), loaded)
            if line is None:
                break
        else:
            about = f" {rl(ACCENT)}· state{rl(RESET)}" if session.state else ""
            try:
                line = input(f"{rl(ACCENT)}{rl(BOLD)}laya{rl(RESET)} {rl(DIM)}[{name}]{rl(RESET)}{about} ▸ ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
        if not line:
            continue
        try:
            if line.startswith("/"):
                if not command(session, line):
                    break
            else:
                # What was drawn while typing is the answer; it is asked again only if it is not for this line.
                seen = live.answered(line) if live else None
                if seen and not session.raw:
                    show_answer(seen["body"]["answers"]["q"], seen["usage"], seen["trip"])
                else:
                    session.line(line)
        except Refused as error:
            print(f"{ERR}✘{RESET} {error}")
        except ValueError:
            print(f"{ERR}✘{RESET} that needs a number; /help shows how the command is written")
        except KeyboardInterrupt:
            print(f"\n{MUTED}  interrupted{RESET}")
    if live:
        live.save()
    elif readline:
        try:
            readline.write_history_file(HISTORY)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8095", help="where the app is (default: %(default)s)")
    ap.add_argument("--model", help="put this model on the MLA first")
    ap.add_argument("--ask", metavar="QUESTION", help="answer one yes-or-no question and exit")
    ap.add_argument("--budget", type=int, default=0, help="token budget for a request")
    ap.add_argument("--json", action="store_true", help="print answers as the API's JSON")
    ap.add_argument("--no-banner", action="store_true", help="do not draw the logo (run.sh has, with the versions)")
    args = ap.parse_args()

    client = Client(args.url)
    session = Session(client, args.budget, args.json)
    interactive = args.ask is None and sys.stdin.isatty()
    try:
        if interactive and not args.no_banner:
            banner(client.url)
        loaded = client.loaded()
        if args.model and args.model != loaded:
            if args.model not in client.info()["models"] and not download(client, args.model):
                sys.exit(1)
            load(client, args.model)
        elif loaded is None and interactive:
            print(f"{MUTED}  No model is on the MLA yet.{RESET}")
            choose_model(client)
        elif interactive:
            print(f"{OK}✔{RESET} {loaded} is on the MLA")
        if interactive:
            repl(session)
        else:                             # --ask, or one question per line of standard input
            for text in [args.ask] if args.ask is not None else (line.strip() for line in sys.stdin):
                if text:
                    session.line(text)
    except Refused as error:
        print(f"{ERR}✘{RESET} {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
