"""Dino Arena: what the dinosaur sees, what it may do, and what it should do.

The game itself runs in the browser (`webapp/static/games.html`). This module is the other
half of the contract: the text the model is shown for a game state, and the action a perfect
player takes. It is used to build the fine-tuning set and to check a compiled model.

The harness describes the nearest obstacle in words: what it is, and whether it is `far`,
arriving `now` (inside the window in which reacting works) or already `passing`. Turning a
distance into that word is arithmetic and is done here; choosing the action is the model's job,
including knowing that a high bird is not to be jumped at.

The physics constants and `JUMP_WINDOWS` must match the ones at the top of games.html.
"""
import json
import math

ACTIONS = ["run", "jump", "duck"]
QUESTION = {"type": "choice", "instructions": "What should the dinosaur do?", "criteria": ACTIONS}

OBSTACLES = ["none", "cactus", "low bird", "high bird"]
TIMINGS = ["far", "now", "passing"]
SPEEDS = {"slow": 10.0, "medium": 13.0, "fast": 16.0}   # cells per second
MAX_DISTANCE = 20         # cells; nothing further away is reported
PASSED_DISTANCE = -1      # an obstacle is dropped from view once it is this far behind

DINO_WIDTH = 0.6          # cells
OBSTACLE_WIDTH = 0.6
JUMP_SECONDS = 0.6
JUMP_HEIGHT = 2.2         # cells, at the apex
CACTUS_HEIGHT = 1.0
BIRD_WINDOW = (0, 3)      # whole cells: a bird this close is arriving now


def jump_window(speed: str) -> tuple[int, int]:
    """Distances (whole cells) at which starting a jump clears a cactus.

    The dinosaur is above the cactus while 4 * H * u * (1 - u) > cactus height, u = t / T. The
    cactus must arrive after that begins and leave before it ends; half a cell of slack on
    each side covers the rounding of the distance.
    """
    v = SPEEDS[speed]
    root = math.sqrt(1.0 - CACTUS_HEIGHT / JUMP_HEIGHT)
    clear_from, clear_to = JUMP_SECONDS * (1 - root) / 2, JUMP_SECONDS * (1 + root) / 2
    nearest = clear_from * v
    farthest = clear_to * v - (DINO_WIDTH + OBSTACLE_WIDTH)
    return math.ceil(nearest + 0.5), math.floor(farthest - 0.5)


JUMP_WINDOWS = {speed: jump_window(speed) for speed in SPEEDS}


def timing(obstacle: str, distance: int, speed: str) -> str:
    lo, hi = JUMP_WINDOWS[speed] if obstacle == "cactus" else BIRD_WINDOW
    return "far" if distance > hi else "now" if distance >= lo else "passing"


def state(obstacle: str, when: str, speed: str) -> dict:
    """The observation, as the JSON object the model reads."""
    return {"obstacle": obstacle, "timing": when, "speed": speed}


def observe(obstacle: str, distance: int, speed: str) -> dict:
    """What the harness reports for an obstacle `distance` whole cells ahead."""
    if obstacle == "none" or distance > MAX_DISTANCE:
        return state("none", "far", speed)
    return state(obstacle, timing(obstacle, distance, speed), speed)


def best_action(observation: dict) -> str:
    obstacle, when = observation["obstacle"], observation["timing"]
    if obstacle == "cactus":
        return "jump" if when == "now" else "run"
    if obstacle == "low bird":
        return "duck" if when in ("now", "passing") else "run"   # stay down until it is gone
    return "run"   # nothing there, or a high bird: jumping would hit it


def all_states() -> list[tuple[dict, str]]:
    """Every observation the game can produce, with the right action."""
    out = []
    for speed in SPEEDS:
        out.append((state("none", "far", speed), "run"))
        for obstacle in OBSTACLES[1:]:
            for when in TIMINGS:
                observation = state(obstacle, when, speed)
                out.append((observation, best_action(observation)))
    return out


if __name__ == "__main__":
    print("jump windows", JUMP_WINDOWS)
    rows = all_states()
    print(len(rows), "states;", {a: sum(1 for _, b in rows if b == a) for a in ACTIONS})
    print(json.dumps(rows[2][0]), "->", rows[2][1])
