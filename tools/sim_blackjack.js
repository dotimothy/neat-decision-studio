// Plays blackjack without a browser or a board, with the game code the page uses
// (webapp/static/blackjack-core.js), assuming the model picks the rule each description is
// meant to select (tools/blackjack_check.py confirms that on the board).
//
//   node tools/sim_blackjack.js [hands]
//
// It answers what remembering the cards is worth: the same game played by fixed basic
// strategy, by the described facts with flat bets, and by the described facts with the bet
// chosen from the count.
const path = require("path");
const g = require(path.join(__dirname, "..", "webapp/static/blackjack-core.js"));

// One hand from `shoe`. `policy` returns {bet, double(cards, up), play(cards, up)}.
function playHand(shoe, policy) {
  if (shoe.needsShuffle) shoe.shuffle();
  const bet = policy.bet(shoe);
  const player = [shoe.draw(), shoe.draw()], dealer = [shoe.draw(), shoe.draw()];
  player.forEach((card) => shoe.remember(card));
  shoe.remember(dealer[0]);
  const upcard = g.cardValue(dealer[0]);
  let stake = 1;
  if (!g.isBlackjack(player) && !g.isBlackjack(dealer)) {
    if (g.canDouble(player) && policy.double(player, upcard, shoe)) {
      stake = 2;
      player.push(shoe.draw()); shoe.remember(player.at(-1));
    } else {
      for (;;) {
        const { total, soft } = g.handValue(player);
        if (total >= 21 || policy.play(total, soft, upcard, shoe) === "stand") break;
        player.push(shoe.draw()); shoe.remember(player.at(-1));
      }
    }
  }
  shoe.remember(dealer[1]);
  if (g.handValue(player).total <= 21 && !g.isBlackjack(player)) {
    while (g.handValue(dealer).total < 17) { dealer.push(shoe.draw()); shoe.remember(dealer.at(-1)); }
  }
  return { units: g.settle(player, dealer) * stake, bet };
}

const basic = {
  bet: () => 1,
  double: (cards, up) => g.basicStrategy(g.handValue(cards).total, false, up, true) === "double",
  play: (total, soft, up) => g.basicStrategy(total, soft, up, false),
};
const described = (remember, countBets) => ({
  bet: (shoe) => countBets ? g.BET_UNITS[g.intendedRule(g.betFacts(shoe.trueCount))] : 1,
  double: (cards, up, shoe) => { const { total, soft } = g.handValue(cards);
    return g.DOUBLE_ACTIONS[g.intendedRule(g.doubleFacts(total, soft, up, remember ? shoe.valueOdds() : g.FRESH_ODDS))] === "double"; },
  play: (total, soft, up, shoe) => g.PLAY_ACTIONS[g.intendedRule(g.playFacts(total, soft, up, remember ? shoe.valueOdds() : g.FRESH_ODDS))],
});

const hands = Number(process.argv[2] || 2000000);
console.log(`${hands.toLocaleString()} hands each, one deck, reshuffled when fewer than 21 cards are left`);
for (const [name, policy] of [
  ["basic strategy, flat bet", basic],
  ["described facts, no memory, flat bet", described(false, false)],
  ["described facts from remembered cards, flat bet", described(true, false)],
  ["the same, bet 1 / 3 / 6 by the count", described(true, true)],
]) {
  const shoe = new g.Shoe(20261005);
  let won = 0, wagered = 0;
  for (let hand = 0; hand < hands; hand++) { const { units, bet } = playHand(shoe, policy); won += units * bet; wagered += bet; }
  console.log(`  ${name.padEnd(48)} ${(100 * won / hands).toFixed(2).padStart(6)} units per 100 hands  (${(100 * won / wagered).toFixed(2)}% of money bet, average bet ${(wagered / hands).toFixed(2)})`);
}
