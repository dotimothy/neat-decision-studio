"""Blackjack: what the model is asked, and why it is asked that way.

Unlike Dino Arena this game uses a general Laya checkpoint as it ships, with no fine-tuning.
The game itself, the memory of the cards and the arithmetic are in
`webapp/static/blackjack-core.js`; this module holds the same questions and descriptions for
`tools/blackjack_check.py`, which runs every one of them through the board. The two must stay
the same.

How the questions came to be put this way (measured on the board with the English model):

* The model does not do arithmetic. Given the totals ("hard 16 against a 10") it gives the
  same answer for every hand: 42-58% agreement with basic strategy, the base rate. So the
  harness turns numbers into facts in words.
* It does not weigh facts against each other. Asked "hit or stand?" about a hand described in
  words, none of 108 wordings got all eight situations right. So the rules of thumb are
  written out as the question's criteria, and the model's job is to recognize which rule the
  described hand falls under.
* Four rules is about what one question holds. A fifth rule for doubling brought it down to
  8 of 10, so doubling and bet size are questions of their own.
* Keep the card totals out of the text: with them in, agreement fell from 98% to 80-91%.

The memory is the harness's, not the model's: every card shown since the shuffle. It enters
through the facts. "Drawing is safe" and "the dealer is likely to bust" are computed from the
cards still unseen, the edge from doubling likewise, and the richness of the remaining cards
(a Hi-Lo count per deck left) decides the bet.
"""

PLAY_QUESTION = {
    "type": "choice",
    "instructions": "Which rule applies to this blackjack hand?",
    "criteria": {
        "stand_pat": "the hand is strong",
        "free_card": "the hand is weak and drawing is safe",
        "let_dealer_bust": "the hand is weak, drawing is risky, and the dealer is likely to bust",
        "must_draw": "the hand is weak, drawing is risky, and the dealer is likely to make a hand",
    },
}
PLAY_ACTIONS = {"stand_pat": "stand", "free_card": "hit", "let_dealer_bust": "stand", "must_draw": "hit"}

DOUBLE_QUESTION = {
    "type": "choice",
    "instructions": "Should I double down?",
    "criteria": {"double_large": "the edge is large", "double_small": "the edge is small",
                 "draw_none": "there is no edge"},
}
DOUBLE_ACTIONS = {"double_large": "double", "double_small": "double", "draw_none": "play on"}

BET_QUESTION = {
    "type": "choice",
    "instructions": "How much should I bet on the next blackjack hand?",
    "criteria": {"small": "ordinary or poor", "medium": "rich", "large": "very rich"},
}
BET_UNITS = {"small": 1, "medium": 3, "large": 6}

ACE = 11                      # the dealer's upcard is 2..10, or 11 for an ace
UPCARDS = range(2, 12)


def play_facts(total: int, soft: bool, upcard: int) -> str:
    """The hit-or-stand description of a hand from a fresh deck.

    A hand is strong from hard 17 or soft 18. From a full deck, drawing is safe for any soft
    hand and for a hard total of 11 or less, and a dealer showing 2-6 busts at least 30% of the
    time. During play the page computes the last two from the cards remembered instead.
    """
    hand = "strong" if total >= (18 if soft else 17) else "weak"
    draw = "safe" if soft or total <= 11 else "risky"
    dealer = "likely to bust" if upcard <= 6 else "likely to make a hand"
    return f"Hand: {hand}. Drawing a card: {draw}. Dealer: {dealer}."


def double_facts(edge: str) -> str:
    """`edge` is large, small or none: the expected gain from taking exactly one card."""
    return f"My edge over the dealer if I take exactly one card: {edge}."


def bet_facts(richness: str) -> str:
    return f"Cards left in the shoe: {richness}."


def descriptions() -> list[tuple[str, dict, dict, str]]:
    """Every distinct text the game can send: (text, question, rule -> action, intended rule)."""
    out = []
    for hand in ("strong", "weak"):
        for draw in ("safe", "risky"):
            for dealer in ("likely to bust", "likely to make a hand"):
                rule = ("stand_pat" if hand == "strong" else "free_card" if draw == "safe"
                        else "let_dealer_bust" if dealer == "likely to bust" else "must_draw")
                out.append((f"Hand: {hand}. Drawing a card: {draw}. Dealer: {dealer}.",
                            PLAY_QUESTION, PLAY_ACTIONS, rule))
    for edge, rule in (("large", "double_large"), ("small", "double_small"), ("none", "draw_none")):
        out.append((double_facts(edge), DOUBLE_QUESTION, DOUBLE_ACTIONS, rule))
    for richness, rule in (("poor in tens and aces", "small"), ("ordinary", "small"),
                           ("rich in tens and aces", "medium"), ("very rich in tens and aces", "large")):
        out.append((bet_facts(richness), BET_QUESTION, {k: f"bet {v}" for k, v in BET_UNITS.items()}, rule))
    return out


def basic_strategy(total: int, soft: bool, upcard: int) -> str:
    """Hit or stand by basic strategy (dealer stands on all 17s; doubling aside)."""
    if soft:
        return "stand" if total >= 19 or (total == 18 and upcard <= 8) else "hit"
    if total >= 17:
        return "stand"
    if total <= 11:
        return "hit"
    if total == 12:
        return "stand" if 4 <= upcard <= 6 else "hit"
    return "stand" if upcard <= 6 else "hit"


def all_hands() -> list[tuple[int, bool, int]]:
    """Every hit-or-stand decision point: hard 5-21 and soft 13-21 against each upcard."""
    hard = [(total, False, up) for total in range(5, 22) for up in UPCARDS]
    soft = [(total, True, up) for total in range(13, 22) for up in UPCARDS]
    return hard + soft


if __name__ == "__main__":
    for text, _, actions, rule in descriptions():
        print(f"{text:74s} -> {rule} = {actions[rule]}")
