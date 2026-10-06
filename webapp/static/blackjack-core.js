// Blackjack with a memory: the game, what the harness remembers, and how it describes a
// situation to the model. Loaded by blackjack.html and by tools/sim_blackjack.js; the texts and
// rule tables are repeated in games/blackjack_policy.py, which explains why they are worded
// this way, and must stay the same there.
//
// The model keeps nothing between requests. The memory is the harness's: every card shown since
// the shuffle. From it the harness works out a few probabilities and states them in words; the
// model picks the rule those words fall under.
"use strict";

const RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const SUITS = ["♠", "♥", "♦", "♣"];
const DECK = 52;
const RESHUFFLE_BELOW = 21;          // cards left; about 60% of the deck is dealt
const rankValue = (rank) => rank === 0 ? 11 : rank >= 9 ? 10 : rank + 1;
const hiLo = (rank) => rank >= 1 && rank <= 5 ? 1 : rank === 0 || rank >= 9 ? -1 : 0;   // 2-6 / 10-A

const PLAY_QUESTION = { type: "choice", instructions: "Which rule applies to this blackjack hand?", criteria: {
  stand_pat: "the hand is strong",
  free_card: "the hand is weak and drawing is safe",
  let_dealer_bust: "the hand is weak, drawing is risky, and the dealer is likely to bust",
  must_draw: "the hand is weak, drawing is risky, and the dealer is likely to make a hand" } };
const PLAY_ACTIONS = { stand_pat: "stand", free_card: "hit", let_dealer_bust: "stand", must_draw: "hit" };

const DOUBLE_QUESTION = { type: "choice", instructions: "Should I double down?", criteria: {
  double_large: "the edge is large", double_small: "the edge is small", draw_none: "there is no edge" } };
const DOUBLE_ACTIONS = { double_large: "double", double_small: "double", draw_none: "play on" };

const BET_QUESTION = { type: "choice", instructions: "How much should I bet on the next blackjack hand?", criteria: {
  small: "ordinary or poor", medium: "rich", large: "very rich" } };
const BET_UNITS = { small: 1, medium: 3, large: 6 };

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// One deck, and the memory of what has been shown from it. A card is remembered when it is
// turned face up, so the dealer's hole card is not known until it is revealed.
class Shoe {
  constructor(seed) { this.random = mulberry32(seed); this.shuffle(); }

  shuffle() {
    this.cards = [];
    for (let suit = 0; suit < 4; suit++) for (let rank = 0; rank < 13; rank++) this.cards.push({ rank, suit });
    for (let i = this.cards.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [this.cards[i], this.cards[j]] = [this.cards[j], this.cards[i]];
    }
    this.position = 0;
    this.seen = new Array(13).fill(0);
    this.running = 0;
    this.remembered = 0;
  }

  get cardsLeft() { return DECK - this.position; }
  get needsShuffle() { return this.cardsLeft < RESHUFFLE_BELOW; }
  draw() { return this.cards[this.position++]; }
  remember(card) { this.seen[card.rank]++; this.running += hiLo(card.rank); this.remembered++; }

  // Hi-Lo count per deck still unseen: above zero, tens and aces are over-represented.
  get trueCount() { return this.running / ((DECK - this.remembered) / DECK); }

  // Chance that the next unseen card has each value (2..10, and 11 for an ace).
  valueOdds() {
    const odds = new Array(12).fill(0), unseen = DECK - this.remembered;
    for (let rank = 0; rank < 13; rank++) odds[rankValue(rank)] += (4 - this.seen[rank]) / unseen;
    return odds;
  }

  // Unseen cards in the three Hi-Lo groups, for display: [low 2-6, middle 7-9, high 10-A].
  groupsLeft() {
    const left = [0, 0, 0];
    for (let rank = 0; rank < 13; rank++) left[hiLo(rank) === 1 ? 0 : hiLo(rank) === 0 ? 1 : 2] += 4 - this.seen[rank];
    return left;
  }
}

const FRESH_ODDS = new Shoe(1).valueOdds();

const cardValue = (card) => rankValue(card.rank);

function handValue(cards) {
  let total = 0, aces = 0;
  for (const card of cards) { total += cardValue(card); if (card.rank === 0) aces++; }
  while (total > 21 && aces) { total -= 10; aces--; }
  return { total, soft: aces > 0 };
}
const isBlackjack = (cards) => cards.length === 2 && handValue(cards).total === 21;

function addCard(total, soft, value) {
  total += value;
  if (value === 11) { if (soft) total -= 10; else soft = true; }
  if (total > 21 && soft) { total -= 10; soft = false; }
  return [total, soft];
}

// Where the dealer ends up from this upcard, given the odds of each card: chances of 17..21
// (indexes 0..4) and of busting (index 5). The dealer stands on every 17 and, since the player
// only acts when the dealer has no blackjack, the hole card cannot complete one.
function dealerOutcomes(upcard, odds) {
  const memo = new Map();
  const from = (total, soft) => {
    if (total > 21) return [0, 0, 0, 0, 0, 1];
    if (total >= 17) { const out = [0, 0, 0, 0, 0, 0]; out[total - 17] = 1; return out; }
    const key = total * 2 + (soft ? 1 : 0);
    if (memo.has(key)) return memo.get(key);
    const out = [0, 0, 0, 0, 0, 0];
    for (let value = 2; value <= 11; value++) {
      if (!odds[value]) continue;
      const next = from(...addCard(total, soft, value));
      for (let i = 0; i < 6; i++) out[i] += odds[value] * next[i];
    }
    memo.set(key, out);
    return out;
  };
  const excluded = upcard === 11 ? 10 : upcard === 10 ? 11 : 0;
  const out = [0, 0, 0, 0, 0, 0], scale = 1 - (excluded ? odds[excluded] : 0);
  for (let value = 2; value <= 11; value++) {
    if (value === excluded || !odds[value]) continue;
    const next = from(...addCard(upcard, upcard === 11, value));
    for (let i = 0; i < 6; i++) out[i] += odds[value] / scale * next[i];
  }
  return out;
}

// Expected units won per unit bet by a finished hand of `total` against those dealer outcomes.
function standValue(total, outcomes) {
  if (total > 21) return -1;
  let value = outcomes[5];
  for (let i = 0; i < 5; i++) value += Math.sign(total - (17 + i)) * outcomes[i];
  return value;
}

function bustChance(total, soft, odds) {
  if (soft) return 0;
  let chance = 0;
  for (let value = 2; value <= 11; value++) if (total + (value === 11 ? 1 : value) > 21) chance += odds[value];
  return chance;
}

// Expected units won per unit bet if the hand takes exactly one more card and stops.
function oneCardEdge(total, soft, upcard, odds) {
  const outcomes = dealerOutcomes(upcard, odds);
  let edge = 0;
  for (let value = 2; value <= 11; value++) if (odds[value]) edge += odds[value] * standValue(addCard(total, soft, value)[0], outcomes);
  return edge;
}

// ---- What the model is shown ---------------------------------------------------------------

const DEALER_BUSTS_OFTEN = 0.30;     // chance of busting from which a dealer is "likely to bust"

// Hit or stand: three facts. A hand is strong from hard 17 or soft 18; drawing is safe when no
// card that may still come can bust it; the dealer's chances come from the cards left.
function playFacts(total, soft, upcard, odds) {
  const hand = total >= (soft ? 18 : 17) ? "strong" : "weak";
  const draw = bustChance(total, soft, odds) === 0 ? "safe" : "risky";
  const dealer = dealerOutcomes(upcard, odds)[5] >= DEALER_BUSTS_OFTEN ? "likely to bust" : "likely to make a hand";
  return `Hand: ${hand}. Drawing a card: ${draw}. Dealer: ${dealer}.`;
}

// Doubling is considered on a two-card hard 9, 10 or 11.
const canDouble = (cards) => { const { total, soft } = handValue(cards); return cards.length === 2 && !soft && total >= 9 && total <= 11; };
const EDGE_LARGE = 0.15, EDGE_SMALL = 0.07;
function doubleFacts(total, soft, upcard, odds) {
  const edge = oneCardEdge(total, soft, upcard, odds);
  return `My edge over the dealer if I take exactly one card: ${edge >= EDGE_LARGE ? "large" : edge >= EDGE_SMALL ? "small" : "none"}.`;
}

const COUNT_VERY_RICH = 4, COUNT_RICH = 2, COUNT_POOR = -2;
function shoeRichness(trueCount) {
  return trueCount >= COUNT_VERY_RICH ? "very rich in tens and aces" : trueCount >= COUNT_RICH ? "rich in tens and aces"
       : trueCount <= COUNT_POOR ? "poor in tens and aces" : "ordinary";
}
const betFacts = (trueCount) => `Cards left in the shoe: ${shoeRichness(trueCount)}.`;

// The rule each description is meant to select; tools/blackjack_check.py confirms the model
// selects it. Used by the simulation and to mark the model's slips on the page.
function intendedRule(text) {
  if (text.startsWith("Cards left")) return text.includes("very rich") ? "large" : text.includes(": rich") ? "medium" : "small";
  if (text.startsWith("My edge")) return text.includes("large") ? "double_large" : text.includes("small") ? "double_small" : "draw_none";
  if (text.includes("Hand: strong")) return "stand_pat";
  if (text.includes("card: safe")) return "free_card";
  return text.includes("likely to bust") ? "let_dealer_bust" : "must_draw";
}

// Basic strategy for this game from a fresh deck, for comparison: double, hit or stand.
function basicStrategy(total, soft, upcard, twoCards) {
  if (twoCards && !soft && (total === 11 || (total === 10 && upcard <= 9) || (total === 9 && upcard >= 3 && upcard <= 6))) return "double";
  if (soft) return total >= 19 || (total === 18 && upcard <= 8) ? "stand" : "hit";
  if (total >= 17) return "stand";
  if (total <= 11) return "hit";
  if (total === 12) return upcard >= 4 && upcard <= 6 ? "stand" : "hit";
  return upcard <= 6 ? "stand" : "hit";
}

// Units won per unit bet (before doubling) by `player` against the dealer's finished hand.
function settle(player, dealer) {
  const mine = handValue(player).total, theirs = handValue(dealer).total;
  if (isBlackjack(player)) return isBlackjack(dealer) ? 0 : 1.5;
  if (isBlackjack(dealer) || mine > 21) return -1;
  if (theirs > 21 || mine > theirs) return 1;
  return mine === theirs ? 0 : -1;
}

if (typeof module !== "undefined") {
  module.exports = { RANKS, SUITS, Shoe, FRESH_ODDS, PLAY_QUESTION, PLAY_ACTIONS, DOUBLE_QUESTION, DOUBLE_ACTIONS, BET_QUESTION, BET_UNITS,
                     cardValue, handValue, isBlackjack, dealerOutcomes, oneCardEdge, bustChance, playFacts, canDouble, doubleFacts,
                     betFacts, shoeRichness, intendedRule, basicStrategy, settle };
}
