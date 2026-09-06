# The backtest that kept telling me I was right

Here are two numbers from the same code, on the same three days of made-up
prices, running the same trading rule.

**+20.77%.** And **−0.07%.**

The only difference between them is one line, and that line does something
so obvious it sounds like a formality: it makes the portfolio hold
*yesterday's* decision instead of today's. Delete it and a strategy that
loses a rounding error becomes a strategy that made twenty-one percent in
three days.

Nothing crashes when you delete it. No warning, no exception, no NaN, no
suspicious-looking chart. You get a smooth upward equity curve and a number
you'd be happy to put on a slide.

That asymmetry is the whole reason this project exists, and I want to state
it plainly before anything else: **every way a backtest can be wrong makes
the results look better.** Peek at tomorrow's price — line goes up. Pick
which stocks to trade after seeing which ones did well — line goes up.
Forget to charge trading fees — line goes up. Nothing in the failure
distribution points down. The bugs are all flattering, and flattering bugs
don't get found, because nobody investigates good news.

So I stopped thinking of this as "build a backtester." Four lines of pandas
is a backtester. What I set out to build was a backtester I couldn't fool
myself with, and almost every decision below follows from that one.

I drew the scope tight on purpose: daily bars, US stocks, flat percentage
costs. No intraday data, no order book, no options, no live trading, no
machine learning. Each of those is its own project, and a weak version of
one would have made everything around it less trustworthy rather than more.

## The week Yahoo stopped talking to scripts

I started with the data, expecting it to be the boring part.

The first real decision was the *shape*, and it quietly determines
everything downstream. Price data can be laid out a few ways: one row per
(date, stock), which is what a database would hand you; a dictionary mapping
each ticker to its own little table; or one wide table with dates going down
and stocks going across.

I picked the wide table, because it makes the central calculation disappear.
If prices are a wide grid of dates × stocks, and I build the trading signals
in *exactly the same shape*, then "what I held times what it returned" is
one multiplication across the entire five-year history. No looping over
days. No looping over stocks.

The dictionary version would have forced a Python loop over tickers into
every single calculation — precisely the thing the engine exists to avoid.
The long format would have needed a group-by on every operation. Both work.
Both are slower and read worse. The price of the wide-table choice is that
everything downstream has to conform to it, and I've kept that rule: when a
strategy doesn't fit the shape, the strategy changes, not the engine.

Then there's a finance detail that will silently wreck you. Raw closing
prices are useless. If a stock splits two-for-one, the price halves
overnight, and a naive return calculation records a 50% loss that never
happened. Same story with dividends. So the loader pulls the split- and
dividend-adjusted close and rescales the other price fields by the same
ratio, so they stay consistent with each other.

And then the downloading fell apart.

The plan was to hit Yahoo Finance's public price API. First the `yfinance`
library's normal calls started failing — its authentication endpoint was
returning HTTP 429, "too many requests," no matter how little I asked for.
Fine, I thought, I'll skip the library and call the price endpoint directly,
since that one doesn't need the auth step.

That worked. Then it stopped working. 429 again, on my home wifi, on my
phone's hotspot, on a different connection entirely. With cookies, without
cookies, with the handshake, without it. Meanwhile I could open
finance.yahoo.com in Chrome and watch the exact data render perfectly.

That's when I understood the situation. Yahoo wasn't rate-limiting *me*. It
was declining to serve scripts at all, while happily serving browsers. And
the modern site doesn't embed price history in the page HTML, so scraping
wasn't a fallback either.

The fix is unglamorous and completely reliable: a small JavaScript file you
paste into the browser's own console on Yahoo's site. Run there, the
requests are indistinguishable from the page's own requests, because they
*are* the page's own requests. It dumps JSON, and the loader reads that
through the same parsing and adjustment code as the direct path — so however
the prices arrived, they get treated identically. One parser, one place to
be wrong.

Then I committed the data to the repo. Around 30MB of Parquet, five years of
daily bars, zero missing values.

I want to defend that as a decision rather than excuse it as a workaround. A
project that claims to be reproducible cannot depend on somebody else's API
still existing. With the data committed, someone cloning this in two years
gets identical numbers on a machine with no internet. The download path is
still there for the day Yahoo relaxes, and it's the single place a different
data vendor would need to be plugged in.

The real lesson was that the data layer is where the unglamorous time goes,
and that "it worked on my machine last Tuesday" is not a data source.

## One line

Back to the two numbers at the top.

A signal computed from today's closing price cannot be traded today. The
closing price doesn't exist until the market has closed; by the time you
know it, the chance to act on it at that price is gone. So the engine holds
yesterday's target:

```python
return weights.shift(self.lag_days).fillna(0.0)
```

To show that isn't decorative I built a deliberately tiny test: two stocks,
three days. Day one, stock A jumps 10%. Day two, stock B jumps 10%. The
strategy is a caricature — it buys whatever just moved. Without the lag it
returns +20.77%, because it is "buying" things it has already watched go up.
With the lag it returns −0.07%, which is the truth: it bought each winner
the day after the winning.

Both numbers are asserted in the test suite, so deleting that line produces
a loud, specific failure rather than a great-looking chart. The engine also
flatly refuses to run with a lag of zero. It isn't offered as an option.

There are two sneakier versions of the same bug.

The first: one position-sizing mode weights stocks by recent volatility, so
a calm stock and a wild one contribute similar risk. But volatility is
computed from prices — so lag the signal while computing the sizing from
today's data and you've leaked the information straight back in through a
side door. The sizing is therefore folded into the weights *before* the lag,
so one `shift()` covers both.

The second is worse, and it took me a while to properly appreciate. I'll get
to it, because it needs a strategy to live in.

Everything else in the engine is in service of not lying. Input validation,
for instance, sounds like defensive boilerplate and isn't: pandas will
cheerfully take two tables whose dates don't quite line up, align them for
you, and fill the gaps with nothing. No error. Just a plausible equity curve
computed from a portfolio that was accidentally sitting in cash a third of
the time. The engine refuses to run rather than guess.

Normalization has a similar trap. Strategies emit *desires* — "+1, long
this," "−1, short this," "0, nothing" — and the engine turns those into
weights that sum to a fully-invested portfolio. I divide by the sum of the
*absolute* values, not the plain sum, because in a portfolio with balanced
longs and shorts the plain sum is zero and you've just divided by it. Using
absolute values means a market-neutral book comes out at 100% invested and
0% net exposure, which is what "market neutral" is supposed to mean.

And the portfolio calculation itself, after all that, is one expression:

```python
gross_returns = (positions * asset_returns).sum(axis=1)
```

Whole history, both dimensions, one line. That's the payoff for the
data-shape decision made hours earlier.

There are zero `for` loops in that file. Which creates a problem: if someone
added one later, the numbers would be *identical*. Nothing would break, no
test would notice. So there's a test that parses the file itself and fails
if a loop appears anywhere in it. "Vectorized" is a claim about the code, so
I made the test read the code.

## Three ways to fool yourself

All three strategies implement one method — hand it prices, it hands back
desired positions — and the engine genuinely doesn't know which one it's
running.

**Momentum** is the oldest documented pattern in stocks: things going up
tend to keep going up for a few months. Every day I measure each stock's
return over the last 126 trading days, rank every stock against the others
*on that day*, and buy the top 10%.

"Against the others" is doing a lot of work. The comparison is
stock-versus-stock on the same day, never stock-versus-its-own-past. In a
market where everything fell 30%, this still holds the names that fell
least — a bet on relative strength, which is a very different and much more
hedgeable claim than "this will go up." There's a bug I nearly wrote here:
ranking along the wrong axis. Rank across the row and you compare stocks to
each other today. Rank down the column and you compare today's momentum to
last year's, which is a completely different strategy that runs without
complaint.

**Mean reversion** is the opposite bet on a shorter horizon: a stock that
has shot away from its own recent average tends to snap back. Both effects
are real; they just live on different timescales, which is why you can run
them side by side.

The measure is how many standard deviations the price sits from its 20-day
average. If a stock normally trades near $100 and typically wiggles $2, then
$104 is two sigma rich and $97 is one and a half cheap. Dividing by the
wiggle is what makes it comparable across stocks — a $4 move in a sleepy
utility and a $4 move in a volatile tech name are not the same event, but
"two sigma" means the same thing in both.

The sign is easy to get backwards, and getting it backwards produces a
strategy that still runs: a *high* score means expensive, which means
**short** it. The tests pin the direction down for that reason.

One detail I'm happy with: the threshold to open a position and the
threshold to close it are different — open at 1.0 sigma, close only once it
comes back inside 0.25. With a single threshold, a stock hovering on the
line flips in and out every other day, and since the engine charges fees on
every change, that flickering is expensive. The gap is a deliberate brake.

**Pairs trading** is where the interesting failure lives. Some stocks are
economically tied together — two banks, two refiners — so while each wanders
unpredictably alone, the *gap* between them stays fairly stable. When the
gap stretches unusually wide you bet on it closing: buy the laggard, short
the one that ran. If it works, it works regardless of what the market does.

The statistical machinery for "these two wander together" is cointegration.
The standard test regresses one stock on the other and asks whether the
leftover residual mean-reverts. That residual is what you actually trade,
and the regression also gives you the hedge ratio — how many shares of one
offset a share of the other, so market exposure cancels.

Now the sneaky look-ahead I promised.

The obvious way to build this is: test every possible pair over the whole
five years, find the most cointegrated one, then backtest that pair over
those same five years. It produces a gorgeous chart. It is also completely
worthless, because you *chose the pair using knowledge of how it turned
out.* Nobody could have known in January 2019 which pair would stay glued
together through 2023.

What makes this nastier than a missing `shift()` is that every single day's
arithmetic is still correct. Nothing in the daily calculation looks wrong,
because nothing in the daily calculation *is* wrong. The leak is entirely in
the choice of what to trade. It's look-ahead bias wearing a lab coat.

The fix is a formation window: the first 252 days are used only to pick the
pair and fit the hedge ratio, and no trading happens during them. Everything
after is honest with respect to that choice.

## Grading my own homework

Turning returns into comparable numbers has its own quiet traps.
Annualizing uses 252 days rather than 365, because the data only has entries
for days the market was open. Sharpe and Sortino are computed arithmetically
while the headline return is computed geometrically, because compounding is
what actually happened to the money — mixing the two conventions gives you a
Sharpe that disagrees with everyone else's.

My favourite small one: infinite Sharpe ratios. You get one by dividing by a
standard deviation of zero, which happens any time a strategy didn't trade
during the window you're measuring — constantly, once you start slicing
history into pieces. Every ratio here returns "undefined" instead of
infinity. Infinity would quietly top a leaderboard and poison any average it
landed in. Undefined is the truthful answer.

But the number that actually matters comes from walk-forward validation, and
this is the part that changed my conclusions.

A single backtest over five years tells you how a strategy did on *one*
sample of history — and every choice I made while building it was made by
someone who already knew what happened in that history. That's me. I picked
a six-month momentum window and a 20-day reversion window and a 1.0-sigma
threshold, and even if I never explicitly tuned them, I picked them with a
general sense of what has worked in markets I've read about. The backtest
can't see that bias. It's there anyway.

Walk-forward attacks it by repeatedly splitting time: train on two years,
test on the following six months, slide forward, repeat. Five folds. Stitch
the test windows together and that's an out-of-sample track record — a far
better guess at real performance than any full-sample number.

The gap between the two is the real output. A strategy that scores 2.5
in-sample and 0.1 out-of-sample is not a good strategy that got unlucky.
It's a curve fit.

One implementation detail does most of the work: each fold hands the
strategy *only* that fold's slice of prices, never the full history.
Momentum needs its 126 days of warm-up inside the training window so it's
ready when the test window starts — and pairs picks its pair from the first
chunk of whatever you hand it, so slicing per fold means the pair gets
chosen inside the training period. Hand it the whole panel and the pair is
frozen at 2019 forever, and you've leaked the thing you built the formation
window to prevent.

## Then I stopped picking the stocks

For most of this project the universe was forty large caps I'd chosen by
hand — a few from each sector, all names I recognized. It seemed like a
reasonable sample. Here's what the engine said about it:

| strategy | ann return | ann vol | Sharpe | **out-of-sample Sharpe** | beta |
|---|---:|---:|---:|---:|---:|
| mean reversion | −13.33% | 18.05% | −0.70 | **−0.89** | 0.39 |
| momentum | 19.40% | 27.45% | 0.78 | **0.90** | 0.88 |
| pairs | 2.80% | 5.82% | 0.50 | **−0.25** | −0.02 |
| *SPY buy & hold* | *15.60%* | *20.99%* | *0.80* | — | *1.00* |

Momentum made 19.4% a year against SPY's 15.6%. More money than the index.
If I'd stopped at that column I'd have written this up as a win.

What saved me was reading one column across. It earned that with 27.5%
volatility against SPY's 21.0%, so per unit of risk it was already *worse*
than just buying the index — and a beta of 0.88 said most of the return was
market exposure anyone can buy for free. The honest description was "a
leveraged index fund with extra trading costs."

Then I replaced my forty names with the S&P 500.

I want to be precise about why, because it wasn't tidiness. Ranking forty
stocks against each other is a thin cross-section — thin enough that the
result can hinge on a handful of names. And I picked those names. A
cross-sectional strategy evaluated on a universe its author chose is partly
a measurement of the author's taste, and I had no way to tell how much.

476 of the 503 current constituents have complete price history over the
window. Nothing about the code changed. Every strategy got worse.

| strategy | ann return | ann vol | Sharpe | max DD | **out-of-sample Sharpe** | beta |
|---|---:|---:|---:|---:|---:|---:|
| mean reversion | −8.03% | 17.83% | −0.38 | −38.2% | **−0.20** | 0.39 |
| momentum | 13.57% | 22.26% | 0.68 | −34.0% | **0.77** | 0.87 |
| pairs | 1.55% | 16.33% | 0.18 | −19.4% | **−0.18** | 0.10 |
| *SPY buy & hold* | *15.60%* | *20.99%* | *0.80* | *−33.7%* | — | *1.00* |

Momentum no longer even wins on the raw number: 13.57% against the same
15.6%. Its information ratio against SPY is −0.11, meaning the active part
isn't out-performance at all, it's noise with a fee attached. The thin
cross-section had been carrying it, and I only found that out by taking away
my own ability to choose.

Pairs is where widening the universe turned a decent lesson into a much
sharper one. Forty stocks give you 780 possible pairs. 476 give you 113,050.
The best of those — two regional banks — cointegrates at p = 0.0007, a far
more convincing number than anything the small universe could offer. Full
sample it scores 0.18. Out of sample it scores **−0.18**.

So searching 145 times harder bought a much better-looking pair and exactly
zero edge. That's the cleanest statement of the problem I could have asked
for, and it's really a statement about multiple testing: at a 5% threshold
across 113,050 tests you'd expect something like 5,600 pairs to look
significant by pure chance. "It passed a cointegration test" carries almost
no information unless the pair was chosen before the period you score it on.

Mean reversion isn't wrong so much as eaten alive. It trades 478,302 times,
about 3.6% of capital a year in fees before it has been right or wrong about
anything. Turn the cost model off and it looks respectable. That is the
entire argument for having a cost model.

Three strategies, three different reasons for not working: one is beta in
disguise, one is selection bias, one is transaction costs.

I should be honest about which direction the remaining bias runs. Those 476
names are the S&P 500 *as it stands today* — everything that went bankrupt
or got acquired or shrank out of the index between 2019 and 2023 is missing
entirely, and 118 of the names in it only joined the index after 2019, so
holding them from the start is a trade nobody could have made. Widening the
universe fixed my selection bias and made survivorship bias worse. Both
errors flatter the results, including the benchmark.

## Building the same thing twice

At this point it worked. I built a second, completely different engine
anyway, and this is the part I'd most want to be asked about.

The first engine thinks in weights and computes the whole history in a few
whole-table operations. The second one *simulates*: one day at a time,
holding an actual cash balance and an actual share count for each stock,
placing orders that fill at the closing price. Bar arrives, state updates,
orders go out.

The point wasn't redundancy. It was to find out what the fast one was
quietly assuming, because fast and wrong is worse than slow.

One design decision makes the comparison mean anything: the event-driven
engine *calls the vectorized engine's weight calculation*. They cannot
disagree about what to hold. So every difference between their outputs is
attributable to *how* it gets held, rather than being two different
portfolios doing two different things — which would have told me nothing.

With fees off, the two agree to about fifteen decimal places over 1,258
days. That's the assertion I care about, because gross return has no
modelling choices in it. If it diverged, one of them would have a bug in the
lag or the weighting.

Getting there meant catching an off-by-one I nearly shipped.

The obvious thing to write in a simulation loop is "today, fetch the target
from `lag_days` ago." That's wrong, and it took me a while to see why. If
orders fill at *today's* close, the position you just established is exposed
to *tomorrow's* move — the simulation is already lagged by a day,
structurally, just from how it works. So a configured lag of 1 has to map to
*zero* extra delay in the loop.

Write the obvious version and you double-lag the entire portfolio, forever.
Nothing crashes. The equity curve looks fine. The two engines just quietly
disagree by one day for eternity. I caught it by reasoning through the
convention on the vectorized side before writing any tests, and there's now
a test pinning the two engines' positions together so it can't drift back.

And then the comparison exposed something I did not predict.

The event-driven engine reports *more* trading than the vectorized one, for
every strategy. The reason is a blind spot rather than a bug.

The vectorized engine measures trading as the change in *target weights*.
So if a strategy says "hold 5% of AAPL" every day forever, it measures zero
trading after the initial purchase. That's just not true. Holding a constant
*weight* is not holding a constant *position* — prices move overnight, your
5% drifts to 5.3%, and pulling it back is a real trade a real broker really
charges you for. The simulation tracks shares, so it sees those trades. The
vectorized version structurally cannot.

This bites hardest on pairs, where measured trading jumps 40%. Which makes
sense once you see it: pairs holds a nearly static two-stock position, so
almost *all* of its real trading is drift correction — exactly the category
the fast engine is blind to. The strategy that looked cheapest to trade is
the one whose costs were most understated. That finding fell out of the
reconciliation; it wasn't something I went looking for.

None of it reverses a conclusion. What changed is that I now know the fast
engine's cost estimate is biased optimistic, and biased *more* for
low-turnover strategies than high-turnover ones — the opposite of what I'd
have guessed.

The broader thing I took away: real trading systems are event-driven not
because they're more accurate, but because in production your backtest and
your live trader have to be the *same code*. A live trader is inherently an
event loop. A vectorized backtest can't be run live at all — it needs the
entire future sitting in a table before it can compute anything. It's the
right tool for research throughput, and it will lie to you about execution,
quietly, in the flattering direction.

## The C++ part, and why its best number got smaller

I wanted a compiled component, but not a fake one, so the first question was
what to port.

The tempting target is the vectorized engine's main calculation. That would
have been a wasted weekend: it's already NumPy, which is already compiled C
with vector instructions underneath. An honest measurement would have come
back at roughly 1x and I'd have had a chart showing my C++ tied with
somebody else's C.

The simulation loop is the opposite situation, and this is why building it
first mattered. It's irreducibly serial — tomorrow's equity depends on
today's fills — and in Python each day pays for about fifteen separate NumPy
calls. The overhead of each call (allocate a temporary, check types, manage
reference counts, return an object) is *fixed*, so how much of it there is
to delete depends entirely on how much real arithmetic sits underneath.

So the C++ file is a line-for-line translation of the Python loop. Same
variable names, same order of operations. Deliberately boring, because I
want to measure the language, not compare two algorithms.

Correctness came before timing, since a speedup from code that computes
something else isn't a speedup. Across all three strategies the largest
disagreement in daily returns is about 7e-14, with identical trade counts —
floating-point ordering noise, because NumPy sums in pairs and my loop sums
in order, so they differ in the last bit.

The loop came out 3 to 25 times faster depending on the strategy, and 1.5 to
10x end to end — I report both, because a user calls the full function,
which also builds a dozen pandas objects around the loop that C++ never
touches.

Those numbers used to be bigger. On the forty-stock universe the same
benchmark said 28 to 47 times on the loop and 5 to 10 end to end. Widening
to 476 stocks cut it by roughly a factor of three, without either
implementation changing by a single line.

That sounds like bad news and is actually the most satisfying result in the
project, because I'd already predicted it. Here's the measurement that
explains it — same number of days, varying only how many stocks are in the
portfolio:

```
  assets      python       cpp   speedup   python us/bar   cpp us/bar
       5      9.65ms    0.06ms    161.8x            7.67         0.05
      20     17.38ms    0.38ms     45.6x           13.81         0.30
      40     28.94ms    0.70ms     41.5x           23.01         0.55
     100     61.23ms    1.91ms     32.0x           48.67         1.52
     500    316.08ms   11.94ms     26.5x          251.26         9.49
```

Read the two right-hand columns. Going from 5 stocks to 500, the Python loop
gets about 33 times slower per day while the C++ loop gets 190 times slower.
C++ was only ever paying for the arithmetic, and the arithmetic is the part
that grew. Python pays the same fixed dispatch cost whether the array holds
5 elements or 500, so the narrower the portfolio, the more of the runtime is
pure interpreter tax and the more there is to delete.

So the claim was never "C++ beats NumPy at maths." It's "C++ doesn't pay a
dispatch tax fifteen times a day." A benchmark that just said "47x faster!"
would have hidden the entire mechanism — and would have quietly become a lie
the moment I changed the universe. A speedup that survives changing the
problem size without explanation is a speedup you don't understand.

Two things had to change to make the port work, and both turned out to be
improvements to the *Python* side.

The first was a genuine design flaw the port surfaced. My original loop
counted an order as real if the change in share count exceeded a tiny
absolute threshold. That's not scale-free — a sliver of a share of a $1
stock and of a $1,000 stock are not the same event — and it broke.
Specifically: hold a single stock at 100% and the target share count is
*exactly* what you already own, so every day's difference is floating-point
dust sitting right on the threshold, tipping either way depending on
summation order. The two engines disagreed on the trade count by one. The
diagnosis matters more than the fix: this wasn't a porting bug, it was a bug
in code I'd already written and tested, which the port dragged into the
light. Measuring the order as a *fraction of the portfolio* puts that dust
three orders of magnitude clear of the line.

The second was subtler. My equivalence tests failed on the cash balance
differing by 5e-10, and my first instinct was that I'd mistranslated
something. I hadn't. Cash is computed as a residual — total equity minus
everything you're holding — and in a long/short portfolio those are two
numbers around $2 million that nearly cancel to a balance of a few dollars.
Cash keeps absolute precision at the scale of its *inputs* while its
relative precision is destroyed. Textbook catastrophic cancellation,
behaving exactly as advertised. The tests now compare dollar amounts in
dollars and ratios relatively, because one tolerance for both would either
fail on cash forever or wave through a real divergence in returns.

The loop also releases Python's global interpreter lock, so two backtests
can genuinely run on two threads. There's a test that asserts this by
*timing* it, because if someone removed the release, nothing else in the
suite would notice.

## What the tests are actually for

There are 106 and they run in about three seconds, but the count isn't the
point. Every one exists because some specific wrong answer would otherwise
have looked right.

Some check arithmetic against numbers I worked out by hand. Some exist to
catch a silent lie — the toy dataset asserting both +20.77% and −0.07%, or
the one that rewrites the *future* of a price series and demands every past
signal come back byte-identical, which is a direct assault on look-ahead
that no amount of reading the code substitutes for.

Some check things nothing else could see: the loop-detector that parses the
engine's own source, the GIL-release timing test. Both guard properties
where the wrong behaviour produces identical output.

One pair encodes a decision rather than a behaviour: the two engines must
match *exactly* on gross returns with fees off, and may differ slightly with
fees on. That's my modelling stance written down as an executable statement.
They may disagree about cost, never about what was held.

The rest are degenerate cases. A strategy that never trades. A window longer
than the data. A flat-lined stock with zero volatility. A universe where no
pair cointegrates. Every one is a place where a reasonable-looking
implementation divides by zero and poisons something downstream.

## What I'd tell someone starting one of these

The whole thing ships as a Docker image that reproduces the tables, the
engine comparison and every chart with no arguments and no network. Every
number above is generated by the code that computes it.

The limitations I'd rather state than have someone find: survivorship bias,
described earlier and worse than it looks. Costs are a flat percentage with
no market impact. Shorts are assumed free to borrow, which they aren't,
especially for the names mean reversion wants to short. Sharpe uses a 0%
risk-free rate over a period when rates went from 2.4% to 5%, which makes
every risk-adjusted number here optimistic.

If I kept going, the first change would be re-selecting the pairs on a
rolling basis instead of once, because I still don't know whether pairs
failed because the idea is bad or because one pair decoupled. Then a
survivorship-free universe. Then market impact in the cost model, which
would penalize exactly the strategies that currently look cheapest.

Three things ate most of the time, and all three turned out to be the actual
content rather than obstacles to it. Getting the data, because a source you
don't control can just stop working. Proving there's no look-ahead, because
you can't do that by reading code — you have to produce a number that moves
by 21% when you delete one line. And building the engine twice, because the
second implementation is the only way to find out what the first one was
assuming.

The conclusion I'd defend hardest is the boring one: nothing here beats
buying the index. And I want to be clear that a much better-looking version
of this project was available the entire time, three one-line edits away —
delete the lag, pick the pair on the full sample, turn the fees off. Each
one would have raised the headline number. None of them would have raised an
error.

Most of the work was in not making them.

---

*Code: [github.com/namesarnav/backtesting-python](https://github.com/namesarnav/backtesting-python)*
