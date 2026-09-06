# Building a vectorized backtesting engine

This is a writeup of a daily-frequency portfolio backtesting engine for US
equities: what it does, how it is structured, why the structure is what it
is, and what it produced. The code is Python with an optional C++ extension.

The headline result is negative. Three classical strategies — cross-sectional
momentum, mean reversion, and cointegration-based pairs trading — were tested
on 476 S&P 500 constituents over 2019–2023. None beats a passive SPY position
on a risk-adjusted basis, and one of them only looks profitable until it is
validated out of sample.

## The problem

The arithmetic of a backtest is trivial: multiply positions by asset returns,
sum across assets, compound. Four lines of pandas produce an equity curve.

The difficulty is that the result is unverifiable by inspection. A backtest
has no ground truth to compare against, so a bug does not announce itself —
it produces a number, and every number looks equally plausible.

Worse, the error distribution is one-sided. The common defects all bias
results upward:

- **Look-ahead bias** — trading on information not available at decision
  time. Returns increase.
- **Selection bias** — choosing instruments after observing their
  performance. Returns increase.
- **Omitted transaction costs** — ignoring commission, slippage, and
  turnover. Returns increase.
- **Survivorship bias** — testing on a universe filtered by present-day
  existence. Returns increase.

None of these raises an exception, and none produces an implausible chart.
An implementation containing all four runs cleanly and reports an excellent
Sharpe ratio. The engineering problem is therefore not "compute returns"
but "make each of these failure modes detectable," which means each needs a
specific mechanism and each mechanism needs a test that fails when it is
removed.

Scope was fixed deliberately: daily bars, US equities, flat basis-point cost
model. No intraday data, order book modelling, options, live trading, or
machine learning.

## Data layer

### Panel shape

The most consequential early decision is the in-memory layout of price data.
Three options were considered:

| layout | structure | cost |
|---|---|---|
| long | one row per (date, ticker) | group-by on every operation |
| dict of frames | ticker → DataFrame | Python loop over tickers everywhere |
| wide panel | dates × tickers | must be enforced everywhere |

The engine uses the wide panel: a `DataFrame` with a `DatetimeIndex` and
MultiIndex columns `(field, ticker)`, so `panel["close"]` yields a plain
`(date × ticker)` frame. Signal panels are produced in the same shape.

The reason is that it collapses the core computation to a single expression
over the whole history:

```python
gross_returns = (positions * asset_returns).sum(axis=1)
```

Both other layouts require iteration or reshaping at every step. The cost of
the wide panel is that it must be enforced: when a strategy does not fit the
shape, the strategy changes rather than the engine.

### Price adjustment

Raw closing prices are unusable for return computation. A 2-for-1 split
halves the quoted price overnight and registers as a −50% return. Dividends
produce a smaller version of the same error.

The loader fetches the split- and dividend-adjusted close and rescales
open/high/low by the per-bar ratio `adjclose / close` so all price fields
remain mutually consistent. Volume is left unadjusted, per convention.

### Caching

Two tiers, solving different problems:

- **Per-ticker Parquet**, keyed by ticker, date range, and interval. Adding
  one ticker to the universe re-fetches one ticker, not the whole universe.
- **Panel-level Parquet**, keyed by a hash of (tickers, date range,
  interval, coverage policy). Avoids re-concatenating on every run.

Parquet rather than CSV preserves dtypes and avoids re-parsing date strings.

### Source

Yahoo Finance's chart API is the source. During development, `yfinance`'s
crumb-authentication endpoint returned HTTP 429 independent of request
volume, so the chart endpoint — which requires no crumb — is called directly.

That path later stopped working as well: `query1`/`query2` return 429 to
scripted HTTP clients on every network tested, with or without cookies,
while `finance.yahoo.com` serves the same data to a browser. The site no
longer embeds price history in page HTML, so scraping is not an alternative.

The working route is a JavaScript file executed in the browser console on
Yahoo's origin, where requests are indistinguishable from the page's own.
It writes JSON, and `ingest_browser_panel()` reads it through the *same*
parsing and adjustment code as the direct path, so provenance does not
affect the resulting prices. The capture is chunked at 50 tickers per file
and resumable, since the full payload exceeds 100MB.

The resulting cache — roughly 30MB of Parquet, 1,258 trading days, zero
NaNs — is committed to the repository. This is deliberate: reproducibility
that depends on a third-party API remaining available is not
reproducibility. A clone runs offline and produces identical numbers.

### Coverage filtering

The universe is the S&P 500's 503 current constituents. Not all of them span
the test window.

The original implementation resolved a ragged panel by trimming to the first
date on which every ticker had data. With 40 long-established large caps
that was a no-op. With the S&P 500 it is not: constituents such as COIN,
ABNB, and PLTR listed in 2020–2021, and a single one of them would silently
move the panel's start date forward — converting a five-year backtest into a
two-year one with no error and no warning.

The engine now drops the *ticker* rather than the window, and reports every
drop by name. 27 of 503 are removed: 21 listed after the start date, 6 trade
under symbols that did not exist before 2024. A floor at 75% of the
requested universe raises an error rather than allowing a broken cache to
present itself as a coverage result.

This is a selection rule, so it is stated rather than applied silently. It
biases the surviving universe toward companies already listed in 2019.

## Architecture

```
configs/*.yaml          universe, strategy registry, cost/lag/validation params
     |
     v
engine/data_loader.py   fetch -> adjust -> cache -> align
     |
  price panel           dates x (field, ticker)
     |
     +---------> strategies/*.py    generate_signals(panel) -> signal panel
     |                                    |
     +------------------------------------+
     v                                    v
engine/backtest.py                 engine/event_driven.py
  VectorizedBacktester               EventDrivenBacktester
  whole-panel arithmetic             bar loop, explicit cash and shares
     |                                    |
     v                                    v
  BacktestResult  <--- same fields ---> EventDrivenResult
     |
     +--> metrics/performance.py   Sharpe, Sortino, max DD, Calmar, turnover
     +--> metrics/validation.py    walk-forward folds, benchmark comparison
     +--> viz/plots.py             equity, drawdown, rolling Sharpe
```

Four decisions hold this together.

**One panel shape everywhere.** Prices and signals are identically shaped,
so portfolio return is one element-wise multiply and a row sum. `engine/
backtest.py` contains zero `for` loops, and a test parses the file's AST and
fails if any loop or comprehension appears — because a loop added later
would produce identical numbers and no behavioural test would catch it.

**One strategy interface.** Every strategy implements
`generate_signals(price_panel) -> signal_panel`. The engine does not know
which strategy it is running. Adding a fourth is one file and one YAML
block, with no engine change.

**Configuration is not code.** Universe, costs, lag, allocation method, and
walk-forward window sizes live in `configs/*.yaml`.

**Two engines, one allocation step.** The event-driven engine calls the
vectorized engine's `_target_weights`, so the two cannot disagree about what
to hold. Every difference in their output is attributable to how it is held.

### Execution order

The vectorized engine performs, in order:

1. Validate that prices and signals share an index and columns.
2. Convert raw signals to target weights.
3. **Lag the weights by `lag_days`.**
4. Compute turnover from the change in lagged weights.
5. Charge transaction cost and slippage against turnover.
6. Multiply positions by asset returns and sum across assets.
7. Compound into an equity curve.

Step 1 is not boilerplate. pandas aligns mismatched frames silently and
fills gaps with NaN, which yields a plausible equity curve computed from a
portfolio that was accidentally in cash. The engine raises instead.

Step 2 normalizes by the sum of *absolute* weights, not the plain sum. In a
balanced long/short book the plain sum is zero, and dividing by it produces
infinities. Using absolute values gives a market-neutral book 100% gross
exposure and 0% net exposure.

## Look-ahead bias

A signal computed from day *t*'s close cannot be traded on day *t*; that
price does not exist until the market has closed. The engine holds the
previous day's target:

```python
return weights.shift(self.lag_days).fillna(0.0)
```

To quantify the effect, a synthetic panel of two assets over three days is
used. Asset A rises 10% on day 1; asset B rises 10% on day 2. The test
strategy buys whatever just moved.

| configuration | total return |
|---|---:|
| lag removed | +20.77% |
| lag present | −0.07% |

Both values are asserted in the test suite. A configured lag of zero is
rejected outright rather than offered as an option.

Two related cases required separate handling:

**Volatility-based sizing.** One allocation mode weights assets by inverse
recent volatility. Volatility is computed from prices, so sizing from
current data while lagging the signal reintroduces the leak. Sizing is
therefore folded into the weights *before* the shift, so one operation
covers both.

**Pair selection.** Covered under Strategies below; it is a leak in the
choice of instrument rather than in the timing of a trade.

## Strategies

### Cross-sectional momentum

Each day, compute the trailing 126-day return for every asset, rank assets
against each other *on that date*, and hold the top decile.

The ranking axis matters. Ranking across the row compares assets to each
other on the same day. Ranking down the column compares an asset's current
momentum to its own history — a different strategy that runs without error.
The former is a relative-strength bet: in a market where everything falls
30%, it holds the assets that fell least.

### Mean reversion

Compute the z-score of price against its own 20-day moving average.
Normalizing by the rolling standard deviation makes the measure comparable
across assets: a $4 move in a utility and a $4 move in a volatile technology
name are not equivalent events, but two sigma means the same thing in both.

Sign convention: a high z-score means the asset is expensive, which implies
a **short**. The tests pin this direction, since inverting it produces a
strategy that still executes.

Entry and exit thresholds differ — enter at 1.0 sigma, exit only inside
0.25. With a single threshold, an asset hovering at the boundary enters and
exits on alternating days, and since cost is charged on every position
change, that oscillation is expensive. The gap is a deliberate hysteresis
band.

### Pairs trading

Two economically linked assets may each follow a random walk while the
spread between them remains stationary. When the spread widens, the trade is
long the laggard and short the leader.

Selection uses the Engle-Granger test: regress one price series on the
other and test the residual for stationarity. The residual is the traded
spread; the regression slope is the hedge ratio that cancels market
exposure.

476 assets yield 113,050 candidate pairs, which is too many for the full
test, so candidates are pre-screened by absolute return correlation and the
Engle-Granger test is run on the top 20.

**The selection-bias problem.** Testing all 113,050 pairs over the full
sample, selecting the most cointegrated, and then backtesting that pair over
the same period is invalid — the instrument was chosen using knowledge of
the outcome. Every day's arithmetic remains correct, which is what makes
this harder to detect than a timing error: the defect is entirely in the
choice of what to trade.

The mitigation is a formation window. The first 252 days are used only to
select the pair and fit the hedge ratio; no trading occurs during them.

## Evaluation

### Metric conventions

- Annualization uses 252 trading days, not 365.
- Sharpe and Sortino are computed arithmetically (mean daily return over
  daily standard deviation, scaled by √252); headline return is computed
  geometrically. Mixing the two conventions produces a Sharpe that is not
  comparable to published figures.
- Every ratio returns NaN, not infinity, when its denominator is zero. Zero
  standard deviation occurs whenever a strategy does not trade during a
  measurement window, which is common once history is split into folds.
  Infinity would rank first on any leaderboard and corrupt any average
  containing it.

### Walk-forward validation

A full-sample backtest measures performance on one sample of history, chosen
by someone who already knows what happened in it. Parameter choices — a
126-day momentum window, a 20-day reversion window, a 1.0-sigma threshold —
carry that knowledge even when no explicit optimization was performed.

Walk-forward validation splits time repeatedly: train on 504 days, test on
the following 126, slide forward, repeat for 5 folds. Concatenating the test
windows gives an out-of-sample track record.

The gap between in-sample and out-of-sample results is the actual output. A
strategy scoring 2.5 in-sample and 0.1 out-of-sample is not an unlucky good
strategy; it is a curve fit.

One implementation detail carries most of the weight: each fold passes the
strategy only that fold's slice of prices, never the full history. Momentum
needs its 126-day warm-up inside the training window. Pairs selects its pair
from the beginning of whatever data it receives, so per-fold slicing means
selection occurs inside the training period. Passing the full panel would
freeze the pair at 2019 and reintroduce the leak the formation window exists
to prevent.

## Results

476 S&P 500 constituents, 2019-01-02 to 2023-12-29 (1,258 trading days), net
of 5bp transaction cost and 2bp slippage, positions lagged one day.

| strategy | ann return | ann vol | Sharpe | Sortino | max DD | Calmar | **OOS Sharpe** | beta | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean reversion | −8.03% | 17.83% | −0.38 | −0.52 | −38.2% | −0.21 | **−0.20** | 0.39 | 0.206 |
| momentum | 13.57% | 22.26% | 0.68 | 0.95 | −34.0% | 0.40 | **0.77** | 0.87 | 0.160 |
| pairs | 1.55% | 16.33% | 0.18 | 0.26 | −19.4% | 0.08 | **−0.18** | 0.10 | 0.014 |
| *SPY buy & hold* | *15.60%* | *20.99%* | *0.80* | *1.11* | *−33.7%* | *0.46* | — | *1.00* | — |

### Momentum: market exposure, not alpha

Momentum returns 13.57% against SPY's 15.60%, at higher volatility (22.26%
vs 20.99%) and a lower Sharpe (0.68 vs 0.80). Beta of 0.87 indicates most of
the return is market exposure obtainable for free. The information ratio
against SPY is −0.11, so the active component is not out-performance.

This result changed when the universe was widened. On an earlier universe of
40 hand-selected large caps, momentum returned 19.40% against the same
15.60% benchmark, with 27.45% volatility and Sharpe 0.78 — more money than
the index, worse risk-adjusted performance, and defensible as "a leveraged
index fund with extra trading costs."

On 476 names it loses on the raw return as well. No code changed between the
two runs. Ranking 40 assets is a thin cross-section, thin enough for a
handful of constituents to determine the result, and those constituents were
selected by the author. Widening the universe removes that degree of
freedom.

### Pairs: selection bias, quantified

Full-sample Sharpe is 0.18. Out-of-sample Sharpe is −0.18.

The wider universe sharpens this considerably. 40 assets provide 780
candidate pairs; 476 provide 113,050. The best pair found — two regional
banks — cointegrates at p = 0.0007, far stronger than anything available in
the smaller universe, and still produces a negative out-of-sample result.

This is a multiple-testing effect. At a 5% significance threshold across
113,050 tests, approximately 5,600 pairs pass by chance alone. A
cointegration p-value carries little information unless the pair was
selected before the period being scored.

### Mean reversion: transaction costs

The strategy executes 478,302 trades at 0.206 average daily turnover,
consuming roughly 3.6% of capital per year in costs before accounting for
directional accuracy. With the cost model disabled it appears viable. This
is the case for having a cost model.

### Market neutrality

Pairs carries the lowest market exposure — beta 0.10, max drawdown −19.4%
against −34% to −38% elsewhere. It is noticeably less neutral than the
40-asset version (beta −0.02, 5.82% volatility), since a spread between two
regional banks carries sector risk that a two-name portfolio does not
diversify away.

## Engine reconciliation

The event-driven engine was built after the vectorized one was working, to
determine what the vectorized implementation was assuming. It simulates bar
by bar with an explicit cash balance and per-asset share counts, filling
orders at the close.

Because it calls the vectorized engine's weight calculation, the two cannot
disagree about target holdings. With costs disabled they agree on gross
returns to approximately 1e-15 over 1,258 days.

### The execution-delay off-by-one

The intuitive implementation fetches the target from `lag_days` bars ago.
This is wrong. Orders filling at bar *t*'s close establish a position
exposed to bar *t+1*'s move, so the simulation is already lagged by one bar
structurally. A configured `lag_days` of 1 must map to zero additional delay
in the loop:

```
execution_delay = lag_days - 1
```

The naive version double-lags the portfolio for the entire run. Nothing
fails; the equity curve remains plausible; the two engines disagree by one
day indefinitely. A test now pins the two engines' positions together.

### What the comparison exposed

The event-driven engine reports higher turnover than the vectorized engine
for every strategy.

The vectorized engine measures turnover as the change in target *weights*.
A strategy holding a constant 5% weight registers zero turnover after the
initial purchase — but prices move overnight, the position drifts to 5.3%,
and restoring it is a real trade with a real cost. The event-driven engine
tracks shares and observes those trades. The vectorized engine structurally
cannot.

The effect is largest for pairs, where measured turnover rises 40%. Pairs
holds a nearly static two-asset position, so almost all of its real trading
is drift correction — precisely the category the vectorized engine cannot
see. The strategy that appeared cheapest to trade had the most understated
costs.

Two smaller differences were investigated and kept:

- **Cost timing.** The simulation charges cost on the fill bar; the
  vectorized engine charges it on the bar the position takes effect. A
  one-day shift, immaterial to totals.
- **Pre-commission sizing.** Orders are sized on equity before commission,
  because commission is unknown until the order exists. This is live-system
  behaviour and is why the engines agree exactly only when costs are zero;
  with costs on, gross returns differ by approximately 1e-5.

No conclusion reverses. What changed is knowing that the vectorized cost
estimate is biased optimistic, and biased more for low-turnover strategies
than high-turnover ones.

The general point: production systems are event-driven not because they are
more accurate but because the backtest and the live trader must be the same
code. A live trader is inherently an event loop. A vectorized backtest
cannot be run live at all, since it requires the entire future in memory
before computing anything.

## Performance

### Choosing the port target

Porting the vectorized engine's aggregation to C++ would have measured
nothing: it is already NumPy, which is compiled C with vector instructions.
The honest result would have been approximately 1x.

The event-driven loop is the opposite case. It is irreducibly serial — bar
*t+1*'s equity depends on bar *t*'s fills — and each bar performs roughly
fifteen NumPy calls. Per-call overhead (temporary allocation, dtype
checking, reference counting, object construction) is fixed, so the fraction
of runtime it represents depends on how much arithmetic sits underneath it.

The C++ implementation is a line-for-line translation of the Python loop:
same variable names, same operation order. This measures the language rather
than comparing two algorithms.

### Correctness

Across all three strategies on the real panel, the largest disagreement in
daily returns is 6.9e-14, with identical fill counts. This is
floating-point ordering noise — NumPy reduces pairwise, the C++ loop
accumulates sequentially.

### Wall clock

```
WALL CLOCK  (real panel: 1,258 bars x 476 tickers)

strategy              python       cpp   speedup     (full run() end to end)
----------------------------------------------------------------------------
mean_reversion      248.42ms    9.93ms     25.0x         269.8ms ->   27.9ms  (9.7x)
momentum             39.36ms    4.62ms      8.5x          52.2ms ->   16.3ms  (3.2x)
pairs                11.13ms    3.90ms      2.9x          26.4ms ->   17.6ms  (1.5x)
```

Two figures are reported. The loop is what was ported, so 3–25x measures the
port. Callers invoke `run()`, which also constructs pandas objects the C++
never touches, so the end-to-end gain is 1.5–10x.

### Scaling

```
SCALING  (1,258 bars, varying width -- synthetic)

  assets      python       cpp   speedup   python us/bar   cpp us/bar
---------------------------------------------------------------------
       5      9.65ms    0.06ms    161.8x            7.67         0.05
      20     17.38ms    0.38ms     45.6x           13.81         0.30
      40     28.94ms    0.70ms     41.5x           23.01         0.55
     100     61.23ms    1.91ms     32.0x           48.67         1.52
     500    316.08ms   11.94ms     26.5x          251.26         9.49
```

The speedup decreases as the book widens. From 5 to 500 assets the Python
loop becomes ~33x slower per bar while the C++ loop becomes ~190x slower:
C++ pays only for arithmetic, and arithmetic is what scales. Python pays the
same fixed dispatch cost regardless of array size, so a narrow book is
mostly interpreter overhead and there is more of it to remove.

This predicted the real-panel result. On the 40-asset universe the same
benchmark reported 28–47x on the loop and 5–10x end to end; widening to 476
assets reduced it roughly threefold with no implementation change. The claim
is not that C++ outperforms NumPy at arithmetic — it is that C++ does not
pay a dispatch cost fifteen times per bar.

The loop releases the GIL, so backtests can run concurrently on separate
threads. A timing-based test asserts this, since removing the release would
not change any output.

### Two fixes the port forced

**Fill criterion.** The original loop treated an order as real when the
share delta exceeded an absolute threshold of 1e-12. This is not scale-free:
1e-12 shares of a $1 asset and of a $1,000 asset are different events.
Holding one asset at 100%, the target share count equals the current
holding exactly, so each bar's delta is floating-point noise sitting on the
threshold and tipping either way depending on summation order. The engines
disagreed on trade count by one. This was a pre-existing defect that the
port exposed rather than a porting error. Measuring the order as a fraction
of portfolio equity places that noise three orders of magnitude below the
threshold.

**Tolerance selection.** Equivalence tests failed on a cash balance
differing by 5e-10. Cash is computed as a residual — total equity minus
holdings — and in a long/short book those are two values near $2,000,000
that nearly cancel to a few dollars. Absolute precision is preserved at the
scale of the inputs while relative precision is destroyed: catastrophic
cancellation. Tests now compare dollar amounts absolutely and ratios
relatively, since one tolerance for both would either fail permanently on
cash or admit a genuine divergence in returns.

### Trade log construction

Widening the universe exposed a separate performance defect. `_trade_log`
melted three weight frames to long format and merged them, materializing
`dates × tickers` rows three times before discarding nearly all of them.

At 40 tickers this was invisible. At 476 it dominated: ~180ms against ~10ms
of actual backtest arithmetic — and the same ~180ms for the pairs strategy,
whose complete trade log is 36 rows, since the cost was proportional to the
panel rather than to the output.

Taking non-zero coordinates first makes the work proportional to the number
of trades. `run()` fell from 187/211/191ms to 19/27/26ms across the three
strategies, with output asserted identical row-for-row.

## Testing

106 tests, ~3 seconds. The count is not the point; each exists because a
specific wrong answer would otherwise look correct.

- **Hand-computed baselines.** Sharpe, drawdown, and a buy-and-hold
  portfolio verified against manual calculation rather than against the
  implementation.
- **Look-ahead detection.** The toy panel asserts both +20.77% and −0.07%.
  A second test rewrites the *future* of a price series and requires every
  past signal to be byte-identical.
- **Properties invisible to output.** The AST loop check on
  `engine/backtest.py`, and the GIL-release timing test. Both guard
  behaviour whose violation produces identical results.
- **Cross-engine agreement.** The two engines must match exactly on gross
  returns with costs off, and may differ with costs on. This encodes a
  modelling position as an executable assertion: they may disagree about
  cost, never about holdings.
- **Degenerate inputs.** A strategy that never trades; a rolling window
  longer than the sample; a zero-volatility asset; a universe with no
  cointegrating pair; all-zero signals. Each is a division-by-zero
  opportunity.

## Limitations

- **Survivorship bias.** The universe is present-day S&P 500 membership.
  Companies that went bankrupt, were acquired, or left the index between
  2019 and 2023 are absent, and 118 of the 503 constituents joined the index
  after 2019-01-01, so holding them from the start is not a realizable
  position. The coverage filter adds a third tilt toward already-listed
  companies. All three flatter results, including the benchmark. A
  point-in-time membership file is the only real fix.
- **Daily data only.** No intraday prices, order book, or microstructure.
  Fills are assumed at the adjusted close.
- **Flat cost model.** Real costs vary with size, liquidity, and urgency,
  and market impact grows with position size.
- **No borrow costs.** Short positions are assumed freely available and free
  to hold, which is least true for the names mean reversion wants to short.
- **Zero risk-free rate.** Over a period when rates rose from 2.4% to 5%,
  this makes every risk-adjusted figure optimistic.

## Conclusions

Three strategies fail for three distinct reasons: momentum is market
exposure with a fee attached, pairs is selection bias, mean reversion is
transaction costs. None beats holding the index.

The methodological results are more useful than the strategy results:

1. **Universe size changes conclusions.** Widening from 40 hand-selected
   names to 476 index constituents degraded every strategy with no code
   change. A cross-sectional strategy evaluated on an author-selected
   universe partly measures the author.
2. **Search breadth requires correction.** Increasing the pair search 145x
   produced a far stronger in-sample p-value and no out-of-sample edge.
3. **A second implementation finds what the first assumes.** The
   vectorized engine's turnover measurement is blind to weight drift,
   understating pairs' costs by 40% — a result that emerged from
   reconciliation rather than from review.
4. **Performance claims are conditional on problem size.** The C++ speedup
   fell threefold when the universe widened, exactly as the scaling analysis
   predicted. A speedup that survives a change in problem size without
   explanation is not understood.

A substantially better-looking version of this project was available
throughout, three one-line edits away: remove the lag, select the pair on
the full sample, disable costs. Each raises the headline figure. None raises
an error.

---

*Code: [github.com/namesarnav/backtesting-python](https://github.com/namesarnav/backtesting-python)*
