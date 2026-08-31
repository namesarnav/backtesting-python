# I built a backtesting engine twice, on purpose

*Building a vectorized portfolio backtester, then rebuilding it as an event
loop to find out what the fast version was lying about.*

---

The finished thing is one command:

```bash
docker build -t backtest-engine . && docker run --rm backtest-engine
```

That reproduces every number and chart below — no API keys, no network, no
manual steps. Underneath it are two independent backtesting engines, three
classical trading strategies, walk-forward validation, a C++ extension, and
102 tests.

The headline result is negative, and that is the interesting part: **none of
the three strategies beats simply buying SPY on a risk-adjusted basis**, and
one of them only looks profitable until it is validated properly.

This is the long version — what I built, in what order, what each decision
was between, and the three places where I got something wrong and found out.

---

## Table of contents

1. [Why a backtest is easy to build and hard to trust](#1-why-a-backtest-is-easy-to-build-and-hard-to-trust)
2. [Scope: what I decided not to build](#2-scope-what-i-decided-not-to-build)
3. [The data layer, and the week Yahoo said no](#3-the-data-layer-and-the-week-yahoo-said-no)
4. [The vectorized engine](#4-the-vectorized-engine)
5. [Look-ahead bias, and the one line that prevents it](#5-look-ahead-bias-and-the-one-line-that-prevents-it)
6. [Three strategies](#6-three-strategies)
7. [Scoring it honestly: metrics and walk-forward](#7-scoring-it-honestly-metrics-and-walk-forward)
8. [The results, read properly](#8-the-results-read-properly)
9. [Charts that argue rather than decorate](#9-charts-that-argue-rather-than-decorate)
10. [Building the engine a second time](#10-building-the-engine-a-second-time)
11. [The C++ port](#11-the-c-port)
12. [What 102 tests are actually for](#12-what-102-tests-are-actually-for)
13. [Shipping it](#13-shipping-it)
14. [Limitations, stated plainly](#14-limitations-stated-plainly)
15. [What I'd do next](#15-what-id-do-next)

---

## 1. Why a backtest is easy to build and hard to trust

A backtest answers one question: if I had traded this rule over some period
of history, what would have happened?

The arithmetic is trivial. Multiply what you held by what it returned, sum
across your holdings, compound the result. You can write it in four lines of
pandas. That is exactly the problem — the code is so short that it looks
finished long before it is correct, and every one of the ways it can be
wrong makes the results look *better*, never worse:

- **Look-ahead bias.** Trading on information that did not exist yet. One
  missing `shift()` turns a losing strategy into a winner.
- **Selection bias.** Choosing what to trade using the whole history, then
  backtesting over that same history. Every individual day's arithmetic is
  correct; the leak is in the choice.
- **Ignoring costs.** A strategy that trades a lot can look excellent gross
  and be catastrophic net.
- **Overfitting.** Tuning parameters until the curve looks good, then
  reporting that curve as if it were a prediction.

There is no error message for any of these. The code runs, the equity curve
slopes up, and the number at the end is fiction. So the entire project is
organized around the four of them: each gets an explicit mechanism, and each
mechanism gets a test that fails loudly if it is removed.

The one thing I wanted to avoid was building a system that told me what I
wanted to hear.

---

## 2. Scope: what I decided not to build

Before writing anything I fixed the boundaries, because "backtesting engine"
can mean anything from 50 lines to a company.

**In scope:** daily bars, US equities, long/short portfolios of up to a few
dozen names, transaction costs as a flat basis-point model, a fixed
allocation rule, walk-forward validation, and reproducibility as a hard
requirement.

**Out of scope, deliberately:** intraday data, order-book microstructure,
options or futures, live trading, a parameter optimizer, and any form of
machine learning. Each of those is a real project on its own, and bolting a
weak version of one onto this would have made every number less trustworthy,
not more.

Three structural decisions came out of that and never changed:

**Configuration is not code.** The universe, costs, lag, allocation method
and walk-forward windows live in `configs/*.yaml`. Changing the universe or
the cost assumption is editing data, not editing logic. Strategies are
registered in YAML too, so adding a fourth means writing one file and one
config block, with no engine change.

**One panel shape everywhere.** Prices and signals are both wide DataFrames
with dates as rows and tickers as columns.

**One strategy interface.** Every strategy implements
`generate_signals(price_panel) -> signal_panel`, and the engine never learns
which one it is running.

Those last two are what make the whole thing vectorizable, so they are worth
a section each.

---

## 3. The data layer, and the week Yahoo said no

### The panel shape

This is the single most load-bearing decision in the repo. Price data can be
laid out several ways:

| Layout | Shape |
|---|---|
| **Long format** | one row per (date, ticker), columns are fields |
| **Dict of DataFrames** | `{"AAPL": df, "MSFT": df, ...}` |
| **Wide panel** ← chosen | dates as rows, `(field, ticker)` MultiIndex columns |

Long format is what a database gives you and what most tutorials use. It is
the wrong shape here: computing a portfolio return means grouping by date on
every operation, and grouping is slow and reads badly.

A dict of per-ticker frames is the shape a beginner reaches for, and it
forces a Python loop over tickers into every single calculation — the exact
thing the engine exists to avoid.

The wide panel makes the core operation disappear into one expression.
`panel["close"]` slices out a plain (date × ticker) frame; a signal panel is
built to be *the same shape*; so "what I held times what it returned" is a
single element-wise multiply over the entire five-year history, and summing
across columns collapses it to one number per day. No loops, no reshaping,
no grouping.

The cost of this choice is that everything downstream must conform to it. A
strategy that wants to return long-format rows does not get to; it is the
strategy that changes, not the engine.

### Adjusted prices

Raw closing prices are useless for backtesting. A 2-for-1 stock split halves
the price overnight, and a naive return calculation reads that as −50%. So
the loader fetches the split- and dividend-adjusted close alongside raw
OHLC, then scales open/high/low by the per-bar ratio `adjclose / close` —
the same approach `yfinance` uses internally for `auto_adjust`. Volume is
left unadjusted, which is the standard convention.

### Caching, in two tiers

Per-ticker Parquet files under `data/cache/`, plus a combined panel-level
cache keyed by a hash of (tickers, date range, interval).

Two tiers rather than one because they answer different questions. The
per-ticker layer means adding one ticker to the universe re-fetches one
ticker, not forty-one. The panel layer means a repeated run skips
re-concatenating and re-aligning entirely. Parquet rather than CSV because
it round-trips dtypes and a `DatetimeIndex` without parsing, and reads back
an order of magnitude faster.

### Missing data

Small gaps inside a ticker's own history get forward-filled up to a limit.
After that, the panel is trimmed to the first date on which *every* ticker
has data. I did not try to represent pre-IPO history as NaN rows — the panel
simply starts later, and because that trim is visible rather than silent, a
suspicious start date shows up immediately instead of quietly propagating
NaNs into the returns.

The result is a panel with zero NaNs, which is asserted by a test rather
than assumed.

### And then Yahoo blocked me

The plan was: hit Yahoo Finance's public chart API with `requests`, done.

`yfinance`'s high-level calls were the first thing to fail — its
crumb-authentication endpoint was returning HTTP 429 independent of my own
request volume, which is a widely reported issue with that specific
endpoint. Fine: the chart endpoint itself does not need a crumb, so I called
it directly.

That worked, then stopped working. `query1` and `query2` both returned 429
to `requests` and to `curl`, on every network I tried — cellular, hotspot,
residential broadband — with and without session cookies, with and without
the crumb handshake. Meanwhile `finance.yahoo.com` in a browser returned 200
and rendered charts perfectly. Yahoo was not rate-limiting *me*; it was
declining to serve scripted clients at all. And the modern site no longer
embeds price history in the HTML, so scraping the page was not an
alternative either.

The fix is unglamorous and completely reliable: `scripts/yahoo_browser_fetch.js`
is pasted into the browser's own JavaScript console on finance.yahoo.com,
where the requests are indistinguishable from the page's own, and it drops a
JSON file. `DataLoader.ingest_browser_panel()` reads that file and runs it
through **the same parsing and adjustment code path** as the direct fetch —
so prices are identical however they arrived.

Then I committed the cache. All 41 tickers (40 names plus SPY), 2019-01-02
to 2023-12-29, 1,258 trading days, zero NaNs, about 2.8MB of Parquet. Which
turned out to be the right call for a reason beyond the block: **a
reproducible project cannot depend on a third-party API being up.** A clone
of this repo produces the same numbers in 2027 as it does today, on a
machine with no network. The direct-fetch path is still there for the day
Yahoo relaxes, and it is the only place a different data vendor would need
to be swapped in.

The lesson I actually took: the data layer is where the unglamorous time
goes, and "it worked on my machine last Tuesday" is not a data source.

---

## 4. The vectorized engine

`engine/backtest.py` is the core, and it is about 290 lines including a lot
of comment. It takes a close-price panel and a same-shaped signal panel and
returns daily returns, positions, a trade log, turnover, and an equity
curve.

The pipeline, in order — and the order matters:

```
signals
  → validate shapes against prices
  → normalize into target weights   (allocation rule)
  → LAG by one day                  ← the whole ballgame
  → measure turnover from the lagged positions
  → charge costs on that turnover
  → (positions × asset returns).sum(axis=1) − costs
  → compound into an equity curve
```

**Validation first, loudly.** If the signal panel's index or columns do not
match the price panel's exactly, the engine raises rather than proceeding.
This matters more than it sounds: pandas will happily align two mismatched
frames for you, silently filling the gaps with NaN, and you will get a
plausible-looking equity curve computed from a portfolio that was
accidentally flat half the time. Loud failure beats a silent wrong answer.

**Signals are desired exposure, not final weights.** A strategy returns
+1/−1/0, and the engine normalizes by gross exposure so the book is fully
invested. Two allocation rules are supported: equal weight, and volatility
weighting where `w ∝ 1/σ` over a trailing window, so a placid utility and a
volatile tech name contribute comparable risk rather than comparable
dollars.

Normalizing by *gross* exposure (the sum of absolute weights) rather than
net is the detail worth pausing on. In a long/short book the net sum can be
zero, and dividing by it produces infinities. Gross normalization means a
market-neutral book with two longs and two shorts sits at 100% gross, 0%
net — which is what "market neutral" means.

**Costs are charged on turnover**, defined as the sum of absolute position
changes, times (5bp transaction + 2bp slippage). Every basis point of
turnover is a basis point a broker really bills you for, and mean reversion
in particular lives or dies on this line.

**The whole portfolio calculation is one expression:**

```python
gross_returns = (positions * asset_returns).sum(axis=1)
```

That is the payoff for the panel-shape decision back in Phase 1. There are
zero `for` loops in the entire module — and because a slow Python loop would
produce identical numbers, no ordinary test would ever notice one appearing.
So a test parses the module's AST and fails if a `For`, `While`, or
comprehension node shows up. "Vectorized" is a claim about the code, so it
is asserted against the code.

---

## 5. Look-ahead bias, and the one line that prevents it

A signal computed from day *t*'s closing price cannot be traded on day *t* —
that price does not exist until the market has closed. So the engine holds
**yesterday's** target:

```python
return weights.shift(self.lag_days).fillna(0.0)
```

To demonstrate that this is not decorative, here is a three-day, two-stock
toy dataset that lives in the test suite. Prices:

```
          A      B
day0    100     50
day1    110     50      → A +10%, B  0%
day2    110     55      → A   0%, B +10%
```

And a signal that chases whatever just moved — buy A on day 1 because A
jumped, buy B on day 2 because B jumped:

| | total return |
|---|---:|
| lag removed | **+20.77%** |
| lag present | **−0.07%** |

A losing strategy becomes a 21% winner from one missing `shift()`. Nothing
crashes. The equity curve is smooth and beautiful. That gap is enormous by
design, so the tests built on this data fail unmistakably if the lag is ever
removed, and `lag_days = 0` is rejected outright rather than offered as an
option.

Two subtler forms of the same bug get handled explicitly:

**Volatility sizing leaks through the back door.** Vol-weighted allocation
is derived from prices, so if you compute weights from today's volatility
and lag only the *signal*, you have leaked. The vol calculation is folded
into the weights before the lag, so one `shift()` covers both.

**Selection bias is nastier than a missing shift.** The obvious way to build
a pairs strategy is to test all 780 possible pairs over the full five years,
pick the most cointegrated one, and backtest it over those same five years.
That produces a beautiful chart and is worthless — you chose the pair using
knowledge of how it turned out. What makes it nastier than a lag bug is that
every individual day's arithmetic is still correct. The leak is in the
choice of *what to trade*, not in when you traded it. The fix is a formation
window: the first 252 rows are used only to select the pair and fit the
hedge ratio, and no trading happens inside them.

Section 8 shows what that selection bias is worth in Sharpe points.

---

## 6. Three strategies

All three implement the same one-method interface, and the engine cannot
tell them apart.

### Momentum — buy what has been going up

The oldest documented equity anomaly (Jegadeesh & Titman, 1993). Every day,
measure each ticker's trailing 126-day (~6 month) return, rank the tickers
*against each other on that day*, go long the top decile.

Two details make it a portfolio strategy rather than a bet on one stock.
It is **cross-sectional**: the comparison is stock-vs-stock on the same day,
never stock-vs-its-own-history, so in a market where everything fell 30% it
still holds the names that fell least. And the ranking is done **per row** —
`rank(axis=1)`. Ranking down a column instead would compare today's momentum
against last year's, which is a complete change of meaning and a classic
silent bug.

### Mean reversion — fade stocks that have stretched

The opposite bet on a much shorter horizon. Both effects are real; they
operate on different timescales, which is why a portfolio can run them side
by side.

The measure is a z-score: `(price − rolling_mean) / rolling_std` over 20
days. If a stock's 20-day average is $100 and its standard deviation is $2,
then $104 is +2σ (rich → short) and $97 is −1.5σ (stretched cheap → long).
Dividing by σ is what makes the number comparable across tickers.

Note the sign: **high z-score means short**. Getting that backwards turns
this into a slow, bad momentum strategy, which is why a test pins the
direction down.

Entry and exit thresholds differ on purpose — enter at |z| ≥ 1.0, exit only
once |z| ≤ 0.25. A single threshold would have positions flickering on and
off every time the z-score jittered across the line, and since the engine
charges cost on every position change, that churn is expensive. The gap
between the thresholds is a deliberate brake on turnover.

### Pairs trading — trade the gap between two stocks that move together

Some pairs of stocks are economically tied. Each wanders unpredictably, but
the *difference* between them is stable. When the gap stretches, bet on it
closing: buy the laggard, short the leader, and profit on convergence
regardless of market direction.

The statistical name for that is **cointegration**: two series can each be a
random walk while a specific linear combination of them is mean-reverting.
`statsmodels`' Engle-Granger `coint()` regresses one on the other and tests
whether the residual is stationary. That residual is the spread we trade:

```
spread = A − (alpha + beta * B)
```

where `beta` is the hedge ratio — how many units of B offset one unit of A.

A 40-ticker universe has 780 possible pairs and cointegration tests are not
free, so candidates are prescreened by correlation (cheap) and the real test
runs only on the most promising ten. If nothing clears the p-value bar, the
strategy holds nothing — which is the correct answer, not a failure, and
there is a test for it.

**One documented simplification:** the pair is chosen once, from the first
formation window, and held for the whole backtest. A production system would
re-select on a rolling basis, since a pair that cointegrated in 2019 may
have decoupled by 2023. Walk-forward validation is exactly where that
assumption gets tested — and it does not survive.

---

## 7. Scoring it honestly: metrics and walk-forward

### The metrics, and two conventions worth defending

Annualization uses **252 trading days**, not 365 calendar days, because a
return series only has entries for days that traded.

Sharpe and Sortino are computed **arithmetically** — mean daily excess
return over daily standard deviation, scaled by √252 — which is the
industry-standard form. Annualized return is separately **geometric**,
because compounding is what actually happened to your money. The two differ,
and mixing them produces a Sharpe that quietly disagrees with everyone
else's, so they are kept distinct.

On infinities: an infinite Sharpe comes from dividing by a zero standard
deviation, which happens whenever a strategy never traded over the window
being measured — a routine occurrence in walk-forward, where a fold can
easily contain no positions at all. Every ratio returns **NaN rather than
inf** in that case. "Undefined" is the honest answer, and unlike infinity it
will not poison a mean or silently top a leaderboard.

The full set: total and annualized return, annualized volatility, Sharpe,
Sortino, max drawdown, Calmar, win rate, average turnover, plus beta, alpha
and information ratio against the benchmark.

### Walk-forward validation

A single backtest over five years tells you how a strategy did on *one*
sample of history. That is weak evidence, because every choice made while
building it — the lookback, the thresholds, the universe — was made by
someone who had already seen that history. Even with no explicit fitting,
the strategy has been tuned by the developer's own knowledge of what worked.
Including mine.

Walk-forward attacks this by repeatedly splitting time:

```
|------ train 504d ------|-- test 126d --|
          |------ train 504d ------|-- test 126d --|
                    |------ train 504d ------|-- test 126d --|
```

Parameters get chosen on each training window and judged only on the test
window that follows — data the choice never saw. Stitching the test windows
together gives an out-of-sample track record. Five folds, two years of
training, six months of testing.

**The gap between in-sample and out-of-sample is the real output.** A
strategy with IS Sharpe 2.5 and OOS Sharpe 0.1 is not a good strategy that
got unlucky. It is a curve fit.

Two implementation details do the actual work here. Each fold hands the
strategy only `prices[train_start : test_end]` — never the full panel — for
two reasons. Strategies need warm-up: momentum produces nothing for its
first 126 days, so it must see the training window to be warm by the time
the test window starts. And `PairsStrategy` picks its pair from the first
`lookback` rows of whatever it is handed, so slicing per fold puts that
formation window inside the *training* period, making the pair selection
genuinely out-of-sample with respect to the test period. Hand it the full
panel and the pair is fixed at 2019 forever — a leak.

---

## 8. The results, read properly

40 liquid US equities across six sectors, 2019-01-02 to 2023-12-29, net of
7bp of costs, positions lagged one day.

| strategy | ann return | ann vol | Sharpe | Sortino | max DD | Calmar | **OOS Sharpe** | beta | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean reversion | −13.33% | 18.05% | −0.70 | −0.97 | −51.2% | −0.26 | **−0.89** | 0.39 | 0.220 |
| momentum | 19.40% | 27.45% | 0.78 | 1.10 | −32.3% | 0.60 | **0.90** | 0.88 | 0.172 |
| pairs | 2.80% | 5.82% | 0.50 | 0.76 | −11.3% | 0.25 | **−0.25** | −0.02 | 0.013 |
| *SPY buy & hold* | *15.60%* | *20.99%* | *0.80* | *1.11* | *−33.7%* | *0.46* | — | *1.00* | — |

**Momentum earns more than SPY but is not better than SPY.** 19.4% against
15.6% looks like a win until you look one column right: 27.5% volatility
against 21.0%. Sharpe 0.78 versus 0.80 — it is *worse* per unit of risk.
Beta 0.88 says most of that return is plain market exposure that anyone can
buy for free, and the information ratio says the leftover outperformance is
not consistent. The honest summary is "a leveraged index fund with extra
trading costs."

This is the single most common way a student backtest fools its author. The
return number is bigger, so it looks like alpha. It is not alpha; it is
beta, plus volatility, minus fees.

**Pairs trading is the cautionary tale.** Full-sample Sharpe 0.50 looks like
a modest success. Walk-forward Sharpe is **−0.25**. That gap is not noise —
it is precisely the value of having selected GS/JPM as the pair after seeing
the whole period. Re-select the pair on each training window, trade the
following six months blind, and the edge is gone. Any pairs backtest that
does not do this is reporting fiction, and this one would have been too.

**Mean reversion is destroyed by costs, not by being wrong.** It trades
33,751 times at an average daily turnover of 0.22 — roughly 3.9% of capital
per year in fees before it is right or wrong about anything. Delete the cost
model and it looks far more respectable. That is precisely why the cost
model exists.

**Only pairs is genuinely market-neutral** — beta −0.02, max drawdown −11.3%
against everyone else's −32% to −51%. It delivers exactly the risk profile
it advertises. It just does not make money out of sample.

Three strategies, three different ways of not working: one is really beta in
disguise, one is really selection bias, one is really transaction costs. I
would rather report that than a Sharpe of 2.5 I could not defend.

---

## 9. Charts that argue rather than decorate

Three chart types, each answering a question a table cannot.

**Equity curves on a log scale.** Linear axes make later years look more
volatile than earlier ones purely because the numbers are bigger; on a log
scale, equal percentage moves are equal distances, which is what you
actually want to compare. Shaded bands mark the 2020 COVID crash and the
2022 selloff — every series reacts to both, which is a sanity check on the
data as much as a narrative device.

**Equity paired with its drawdown.** The equity curve alone hides how it
felt to hold. Drawdown underneath shows the depth and, more importantly, the
*duration* — the number that determines whether a real person would have
stuck with it.

**Rolling 60-day Sharpe.** A single full-sample Sharpe hides *when* a
strategy worked. At a 60-day window every series here swings between roughly
−6 and +7, which is worth internalizing: short-window Sharpe is extremely
noisy, and a strategy "working" for two months means almost nothing.

Matplotlib renders through the `Agg` backend so the charts generate
identically inside Docker with no display attached, and the tests assert
that figures are closed rather than leaked, that a strategy which never
traded still produces a chart instead of crashing, and that a rolling window
longer than the data does not blow up.

---

## 10. Building the engine a second time

At this point the project worked. I built a second, completely different
engine anyway — and this is the part I would talk about in an interview.

`engine/event_driven.py` computes the same history by *simulating* it: one
bar at a time, holding a cash balance and a share count per ticker, placing
orders that fill at the close. Where the vectorized engine thinks in
weights, this one thinks in shares and dollars.

The point was not redundancy. It was to find out what the fast one was
quietly assuming.

### Make them agree first

The design constraint that makes the comparison informative: the
event-driven engine **calls the vectorized engine's allocation step**. The
two literally cannot disagree about *what to hold*, so every difference in
their output is attributable to *how it gets held*. Without that shared
step, any divergence would be uninterpretable — two different books doing
two different things.

With costs switched off, the two produce **identical gross returns** to
floating-point precision, ~1e-15 over 1,258 days. That is the assertion
worth caring about: gross return is "what I held times what it did" with no
modelling choices in it, so divergence there would mean a bug in the lag or
the allocation, not a difference of opinion.

### The off-by-one that nearly got through

Getting there required fixing a real bug, and it is the kind that never
announces itself.

The obvious line to write in an event loop is `target = weights[t - lag_days]`.
It is wrong. Filling at the close of bar *t* means the position established
today is exposed to *tomorrow's* move — the simulation is **already one bar
lagged, structurally**. So `lag_days = 1` maps to *zero* extra delay, and
the correct index offset is:

```python
execution_delay = lag_days - 1
```

Write the obvious version and the entire book is double-lagged forever.
Nothing crashes, the equity curve still looks plausible, and the two engines
quietly disagree by one day for all time. I caught it by reasoning through
the vectorized convention (`positions[t] = w[t−1]` earns `ret[t]`) before
writing the tests, and `test_positions_match_vectorized` now pins it down.

### What the comparison exposed

```
strategy          gross diff  vec return  evt return  vec turnover  evt turnover  cost gap    fills
---------------------------------------------------------------------------------------------------
mean_reversion       1.8e-05     -51.03%     -51.22%        0.2204        0.2251     0.41%   41,527
momentum             5.2e-05     142.38%     140.77%        0.1721        0.1796     0.66%    4,960
pairs                1.2e-05      14.77%      14.19%        0.0127        0.0185     0.51%      956
```

Event-driven turnover is higher for **every** strategy, and total return is
correspondingly lower. Three mechanisms, in order of how much they matter:

**1. Weight drift is a trade the vectorized engine cannot see.** It measures
turnover as the change in *target* weights, `|w[t] − w[t−1]|`. Hold a
permanently constant signal and that is zero forever — it charges for the
initial buy-in and nothing after.

But holding a constant *weight* is not holding a constant *position*. Prices
move overnight, the weights drift apart, and pulling them back to target is
a real trade that a real broker really bills for. The event-driven engine
tracks shares, so it sees those trades and charges for them.

This is starkest in **pairs**, where measured turnover rises 46%
(0.0127 → 0.0185). Pairs holds a near-static two-leg position, so almost all
of its true trading *is* drift correction — exactly the category the fast
engine is blind to. **The strategy that looked cheapest to trade is the one
whose costs were most understated.** That is the finding, and it is not one
I would have predicted.

**2. Cost timing.** A trade decided at bar *t*'s close is filled at bar
*t*'s close, so cash leaves at *t*. The vectorized engine charges it against
*t+1*, when the position becomes effective. A one-bar shift in the cost
series — immaterial to the total, visible day by day.

**3. Orders are sized on pre-commission equity.** The commission is not
known until the order exists, so the order is sized off current NAV and the
fee comes out after. The book therefore sits a fraction of a basis point
above 100% gross once the fee is paid. This is what a live system does, and
it is why gross returns match *exactly* only at zero cost.

None of these reverses a conclusion — momentum still loses on Sharpe, mean
reversion is still bad, pairs still fails out of sample. What changed is the
confidence interval around the cost estimate, and the knowledge that it is
**biased optimistic in the vectorized engine, by more for low-turnover
strategies than high-turnover ones.**

### What each approach is for

|  | Vectorized | Event-driven |
|---|---|---|
| Unit of thought | weights | shares and cash |
| Speed (40 tickers × 5y) | ~15 ms | ~12–34 ms |
| Whole-share orders | no | yes |
| No-trade band | no | yes |
| Sees weight drift | no | yes |
| Tracks cash | no | yes |
| Could place a live order | no | yes |

The event-driven engine can express two frictions the vectorized one cannot
represent at all — whole-share orders (a fixed rounding error in dollars, so
it dilutes as the account grows, which a test demonstrates directly) and a
rebalance threshold, the standard no-trade band that cuts drift-driven churn
at the price of letting weights wander.

**Why real trading systems are event-driven** is not accuracy — it is
*identity*. In production the backtest and the live trader must be the same
code path, and a live trader is inherently an event loop: a bar arrives,
state updates, orders go out. A vectorized backtest cannot be run live at
all; it needs the whole future in a DataFrame before it can compute
anything. It is the right tool for research throughput, and it will lie to
you about execution, quietly, in the optimistic direction.

---

## 11. The C++ port

### Choosing what to port

The tempting target is the vectorized engine's returns aggregation. Porting
it would have been a waste of a weekend: it is already NumPy, which is
already compiled C with SIMD, so an honest measurement would have come back
at roughly 1x.

The event-driven loop is the opposite case. It is irreducibly serial — bar
*t+1*'s equity depends on bar *t*'s fills — and in Python each bar pays for
roughly fifteen separate NumPy calls on 40-element arrays. At that size the
per-call overhead (allocate a temporary, check dtypes, refcount, return)
dwarfs the ~40 multiply-adds of actual arithmetic.

So the second engine was not decoration in front of the C++ work. It is what
made a real speedup possible to measure at all.

`cpp/event_loop.cpp` is bound with pybind11 and is a **line-for-line**
translation of the Python loop — same variable names, same order of
operations — because the point is to measure the language, not to compare
two different algorithms. The extension is optional: with no compiler the
engine falls back to Python and every number in the repo is unchanged.

### Correctness first

A speedup from code that computes something else is not a speedup. Across
all three strategies on the real panel, the largest disagreement in daily
returns is **7.8e-15**, with identical fill counts. That is float-ordering
noise — NumPy reduces pairwise, the C++ loop accumulates in order, so they
differ in the last bit.

### Results

```
WALL CLOCK  (real panel: 1,258 bars x 40 tickers)

strategy              python       cpp   speedup     (full run() end to end)
----------------------------------------------------------------------------
mean_reversion       25.06ms    0.53ms     47.2x          27.5ms ->    2.7ms  (10.0x)
momentum              9.51ms    0.34ms     27.6x          11.2ms ->    1.7ms  (6.7x)
pairs                 7.80ms    0.24ms     32.3x           9.7ms ->    1.9ms  (5.2x)
```

Two numbers, deliberately. The loop is what was ported, so **28–47x**
measures the port. But a user calls `run()`, which also builds a dozen
pandas objects around the loop — constant work C++ never touches — so the
end-to-end gain is **5–10x**. Quoting only the first would overstate what
anyone actually experiences.

### Why the speedup is what it is

Hold the bar count fixed and vary only the width of the book:

```
SCALING  (1,258 bars, varying width -- synthetic)

  assets      python       cpp   speedup   python us/bar   cpp us/bar
---------------------------------------------------------------------
       5      9.85ms    0.06ms    168.2x            7.83         0.05
      20     17.86ms    0.29ms     61.8x           14.20         0.23
      40     28.25ms    0.70ms     40.4x           22.46         0.56
     100     61.28ms    1.87ms     32.8x           48.72         1.48
     500    311.68ms   14.57ms     21.4x          247.76        11.58
```

The Python loop pays the same fixed dispatch cost per NumPy call whether the
arrays hold 5 elements or 500, so the narrower the book, the more of the
runtime is pure interpreter tax and the more there is to remove. At 500
tickers the arrays are finally large enough that NumPy's own arithmetic
dominates, and the gap narrows to 21x.

The claim is therefore **not** "C++ beats NumPy at arithmetic." It is "C++
does not pay a dispatch tax fifteen times per bar." That distinction is the
entire result, and a benchmark that reported only "47x faster!" would have
hidden it.

### What the port cost

Two things had to change, and both were improvements to the Python side.

**The fill criterion was not scale-free.** The original loop counted an
order as real when the share delta exceeded an absolute `1e-12`. But 1e-12
shares of a $1 stock and of a $1,000 stock are not the same event — and the
port broke on it. Hold a single asset at 100% and the target share count is
*exactly* what you already own, so every bar's delta is float dust sitting
right on the threshold, tipping either way depending on summation order. The
two backends disagreed on the fill count by one.

The diagnosis mattered more than the fix: this was not a porting bug, it was
a **design flaw in the Python engine that the port surfaced**. Measuring the
order as a fraction of equity instead puts that dust at ~1e-15, three orders
of magnitude clear of the line, and makes the criterion stable across
implementations. Both engines changed; the reconciliation table did not.

**Tolerances have to respect cancellation.** Cash is a residual: equity
minus everything held. In a long/short book those are two ~$2M numbers that
nearly cancel to a balance of a few dollars, so cash keeps absolute
precision at the scale of the *inputs* while its relative precision is
destroyed. The first version of the equivalence tests failed on a 5e-10
difference in cash. That is not a bug, it is catastrophic cancellation
behaving exactly as documented. The tests now compare money in dollars and
ratios relatively — one tolerance for both would either fail on cash or wave
through a genuine divergence in the returns.

The loop also releases the GIL, so two backtests can run on two threads.
`test_cpp_backend_releases_the_gil` asserts that by timing it — otherwise
removing the release would break nothing any other test could see.

---

## 12. What 102 tests are actually for

Not coverage. Every one of them exists because a specific wrong answer would
otherwise look right.

**Tests that pin down correctness against hand arithmetic.** A buy-and-hold
portfolio must equal the mean of per-asset daily returns, computed manually.
Sharpe, Sortino, Calmar and max drawdown are each checked against a
hand-computed figure rather than against themselves.

**Tests that would catch a silent lie.** The toy panel returns +20.77%
without the lag and −0.07% with it, and both numbers are asserted. Another
test rewrites the *future* of a price series and demands every past signal
stay byte-identical — a direct assault on look-ahead that no amount of
reading the code can substitute for. `lag_days = 0` is rejected.

**Tests for things no other test could see.** The AST check for `for` loops
in the vectorized engine. The GIL-release timing test. Both guard properties
where the wrong behavior produces identical output.

**Tests that encode a decision, not a behavior.** The two engines must agree
*exactly* on gross returns at zero cost, and are allowed to differ by <1e-4
with costs on. That pair of assertions is the modelling decision written
down: the engines may disagree about cost, never about what was held.

**Adversarial and degenerate cases.** A strategy that never trades. A window
longer than the data. A flat-lined stock with zero standard deviation. A
universe where nothing cointegrates. All-zero signals whose row sum is zero.
Each of these is a place where a plausible implementation divides by zero or
returns `inf` and poisons a downstream mean.

The suite runs in about three seconds, entirely offline, on three Python
versions in CI.

---

## 13. Shipping it

**Docker.** `docker build` then `docker run` reproduces the table, the
reconciliation, and all five charts with no arguments and no network. The
image compiles the C++ extension during build, so the container always takes
the fast path.

**CI on every push.** Lint with Ruff, build the extension (so the
backend-equivalence tests actually execute rather than skipping), run the
102 tests on Python 3.11/3.12/3.13, run the full pipeline end to end, and
separately build and run the Docker image.

**Everything regenerable is regenerated.** The results table, the
reconciliation table, and the benchmark are all written to files by the code
that computes them, so no number in the README or in this post is typed by
hand.

---

## 14. Limitations, stated plainly

A backtest that does not list these is hiding them.

- **Survivorship bias.** The universe is 40 companies listed *today*. Firms
  that went bankrupt or were delisted over 2019–2023 are absent, so every
  strategy here — and the SPY benchmark — is measured on a sample that
  survived by construction. This flatters all results.
- **Daily data only.** No intraday prices, no order book, no
  microstructure. Fills are assumed at the adjusted close.
- **Costs are a flat basis-point model.** Real costs vary with size,
  liquidity and urgency, and market impact grows with position size. A
  strategy trading real size would face worse fills than 7bp.
- **No borrow costs or shorting constraints.** Shorts are assumed freely
  available and free to hold. Neither is true, especially for the names a
  mean-reversion strategy wants to short.
- **No risk-free rate.** Sharpe and Sortino use a 0% hurdle. Rates went from
  ~2.4% to ~5% over this period, so real risk-adjusted numbers are *worse*
  than shown.
- **Pairs re-selection is per fold, not rolling.**
- **Five years is a short sample** — one crash and one bear market. Five
  walk-forward folds catch gross overfitting; they are not statistical
  confidence.
- **A backtest is not live performance.** No latency slippage, no partial
  fills, no outages, no risk limits, no capital constraints.

---

## 15. What I'd do next

**Rolling pair re-selection.** The single change most likely to alter a
conclusion. The pairs strategy's −0.25 out-of-sample Sharpe may be selection
bias, or it may be that GS/JPM decoupled and a rolling formation window
would have moved on.

**A survivorship-bias-free universe.** Point-in-time index constituents
would make every number here more honest and probably all of them worse.

**Market impact in the cost model.** A flat 7bp is a placeholder. Impact
that scales with position size relative to volume would penalize exactly the
strategies that currently look cheapest.

**A borrow-cost model**, which would fall hardest on mean reversion, the
strategy that shorts most aggressively.

---

## What I'd tell someone starting this

The engine was the easy part. Three things ate the time, and all three
turned out to be the actual content:

1. **The data layer**, because a source you do not control can simply stop
   working, and reproducibility is a design constraint, not a nice-to-have.
2. **Proving the absence of look-ahead**, because it cannot be done by
   reading code — it has to be demonstrated with a number that changes by
   21% when you delete one line.
3. **Building the thing twice**, because the second implementation is the
   only way to discover what the first one was assuming. The 46% turnover
   understatement on pairs was not something I suspected and went looking
   for. It fell out of the reconciliation.

And the result I would defend hardest is the one that says nothing here
beats buying the index. It would have been trivially easy to produce a
Sharpe of 2.5 from this same data — delete one `shift()`, pick the pair on
the full sample, turn off the cost model. Every one of those is a line of
code, and every one makes the chart more beautiful.

The work was in making sure none of them was there.

---

*Code: [github.com/namesarnav/backtesting-python](https://github.com/namesarnav/backtesting-python) · MIT*
