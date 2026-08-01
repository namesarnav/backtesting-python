# Vectorized Portfolio Backtesting Engine

A daily-frequency backtesting engine for multi-asset equity strategies, built
to demonstrate the things that actually matter in quantitative development:
**correct vectorized computation, explicit handling of look-ahead bias,
realistic transaction cost modelling, and honest out-of-sample validation.**

The headline finding is a negative one, which is the point: of three
classical strategies tested on five years of real US equity data, **none
beats a passive SPY position on a risk-adjusted basis**, and one of them only
looks profitable until it is validated out of sample.

```bash
docker build -t backtest-engine . && docker run --rm backtest-engine
```

That single command reproduces every number and chart below. No API keys, no
network, no manual steps — the price cache is committed.

---

## Results

40 liquid US equities across six sectors, 2019-01-02 to 2023-12-29 (1,258
trading days), net of 5bp transaction cost and 2bp slippage, positions lagged
one day.

| strategy | ann return | ann vol | Sharpe | Sortino | max DD | Calmar | **OOS Sharpe** | beta | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean reversion | −13.33% | 18.05% | −0.70 | −0.97 | −51.2% | −0.26 | **−0.89** | 0.39 | 0.220 |
| momentum | 19.40% | 27.45% | 0.78 | 1.10 | −32.3% | 0.60 | **0.90** | 0.88 | 0.172 |
| pairs | 2.80% | 5.82% | 0.50 | 0.76 | −11.3% | 0.25 | **−0.25** | −0.02 | 0.013 |
| *SPY buy & hold* | *15.60%* | *20.99%* | *0.80* | *1.11* | *−33.7%* | *0.46* | — | *1.00* | — |

`OOS Sharpe` is from walk-forward validation: 5 folds of 504 training days
followed by 126 out-of-sample days, with parameters and pair selection made
on training data only.

### Reading the table

**Momentum earns more than SPY but is not better than SPY.** 19.4% vs 15.6%
looks like a win until you notice the volatility: 27.5% vs 21.0%. Sharpe 0.78
against SPY's 0.80 — it is *worse* per unit of risk. Beta 0.88 says most of
that return is market exposure anyone can buy for free, and an information
ratio of 0.24 says the leftover out-performance is not consistent. The honest
summary is "a leveraged index fund with extra trading costs."

**Pairs trading is the cautionary tale.** Full-sample Sharpe 0.50 looks like a
modest success. Walk-forward Sharpe is **−0.25**. The gap is not noise: it is
the value of having selected GS/JPM as the pair *after seeing the whole
period*. Re-select the pair on each training window and trade the following
six months blind, and the edge disappears. Any pairs backtest that does not
do this is reporting fiction.

**Mean reversion is destroyed by costs, not by being wrong.** It trades
33,751 times at an average daily turnover of 0.22 — roughly 3.9% of capital
per year in fees alone before it is right or wrong about anything. A backtest
without a cost model would have shown something far more flattering, which is
precisely why the cost model exists.

**Only pairs is genuinely market-neutral** — beta −0.02, max drawdown −11.3%
against everyone else's −32% to −51%. It delivers the risk profile it
advertises. It just does not make money out of sample.

---

## Charts

### All strategies vs benchmark

![Strategy comparison](notebooks/results/strategy_comparison.png)

Equity on a log scale so equal percentage moves are equal distances. Shaded
bands mark the 2020 COVID crash and the 2022 selloff — every series reacts to
both, which is a visual sanity check on the data as much as a narrative.

### Momentum, equity and drawdown

![Momentum equity curve](notebooks/results/equity_momentum.png)

The flat first stretch is the 126-day lookback warming up, not a data gap.
Drawdowns land where they should: −32.3% at COVID, and again through 2022.

### Rolling 60-day Sharpe

![Rolling Sharpe](notebooks/results/rolling_sharpe_60d.png)

A single full-sample Sharpe hides *when* a strategy worked. At a 60-day
window every series swings between roughly −6 and +7, which is worth
internalising: short-window Sharpe is extremely noisy, and a strategy
"working" for two months means almost nothing.

---

## Architecture

```
configs/*.yaml          ← universe, strategy registry, cost/lag/walk-forward params
     │
     ▼
engine/data_loader.py   DataLoader
     │                    fetch → split/dividend adjust → Parquet cache → align
     ▼
  price panel           wide DataFrame, dates × (field, ticker) MultiIndex
     │
     ├──────────────► strategies/*.py      generate_signals(panel) → signal panel
     │                   momentum · mean_reversion · pairs
     ▼                          │
engine/backtest.py  ◄───────────┘
  VectorizedBacktester
     normalize weights → LAG BY ONE DAY → returns → subtract costs
     │
     ▼
  BacktestResult        returns · equity_curve · positions · trades · turnover
     │
     ├──► metrics/performance.py   Sharpe, Sortino, max DD, Calmar, win rate, turnover
     ├──► metrics/validation.py    walk-forward folds, SPY benchmark comparison
     └──► viz/plots.py             equity/drawdown, rolling Sharpe, comparison
```

Four design decisions hold the whole thing together:

**One panel shape everywhere.** Prices and signals are both wide DataFrames
with dates as rows and tickers as columns. Because they are identically
shaped, "what I held times what it returned" is a single element-wise
multiply over the entire history — no loops, no reshaping. An AST check in
the test suite confirms `engine/backtest.py` contains zero `for` loops.

**One strategy interface.** Every strategy implements
`generate_signals(price_panel) -> signal_panel`. The engine has no idea which
one it is running; adding a fourth means writing one file and one YAML block,
with no engine change.

**Configuration is not code.** Universe, costs, lag, allocation method and
walk-forward windows live in `configs/*.yaml`.

**The lag is a single, deliberate, commented line.** See below.

---

## Look-ahead bias

A signal computed from day *t*'s closing price cannot be traded on day *t* —
that price does not exist until the market has closed. The engine therefore
holds *yesterday's* target:

```python
return weights.shift(self.lag_days).fillna(0.0)   # engine/backtest.py
```

To show this is not decorative, here is the same three-day toy dataset run
with and without that line:

| | total return |
|---|---:|
| lag removed | **+20.77%** |
| lag present | **−0.07%** |

A losing strategy becomes a 21% winner from one missing `shift()`. The engine
refuses to run with `lag_days=0` rather than offering it as an option, and
`tests/test_backtest.py` fails loudly if the lag is ever removed.

Two subtler forms are handled too:

- **Volatility-based sizing** derives from prices, so it is folded into the
  weights *before* the lag — otherwise it would leak through the back door
  while the signal itself looked correctly lagged.
- **Pair selection** in `strategies/pairs.py` uses a formation window and
  never trades inside it. Choosing a pair over the full history and then
  backtesting on that same history is selection bias, and it is nastier than
  a missing `shift()` because every individual day's arithmetic still looks
  right. The −0.25 out-of-sample Sharpe above is what that costs.

---

## Running it

### Docker (reproduces everything)

```bash
docker build -t backtest-engine .
docker run --rm backtest-engine
```

To keep the generated charts:

```bash
docker run --rm -v "$PWD/notebooks/results:/app/notebooks/results" backtest-engine
```

### Local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m engine.run          # full pipeline: table + charts
pytest -q                     # 64 tests
python scripts/make_charts.py # charts only
```

### Refreshing the price data

The per-ticker Parquet cache is committed, so this is not normally needed.
Yahoo blocks scripted HTTP clients (429 on every network tested, with or
without cookies or the crumb handshake) while still serving browsers, so the
fetch runs from a browser console:

1. open <https://finance.yahoo.com>
2. paste `scripts/yahoo_browser_fetch.js` into the JavaScript console
3. `mv ~/Downloads/yahoo_panel.json data/`
4. `python -c "from engine.data_loader import DataLoader; DataLoader().ingest_browser_panel('data/yahoo_panel.json')"`

---

## Limitations

Stated plainly, because a backtest that does not list these is hiding them.

- **Survivorship bias.** The universe is 40 companies that are listed *today*.
  Firms that went bankrupt or were delisted over 2019–2023 are absent, so
  every strategy here — and the SPY benchmark — is measured on a sample that
  survived by construction. This flatters all results.
- **Daily data only.** No intraday prices, no order book, no microstructure.
  Fills are assumed at the adjusted close.
- **Costs are a flat basis-point model.** Real costs vary with size, liquidity
  and urgency, and market impact grows with position size. A strategy trading
  size would face worse fills than 7bp.
- **No borrow costs or shorting constraints.** Short positions are assumed
  freely available and free to hold. Neither is true, especially for the names
  a mean-reversion strategy wants to short.
- **No risk-free rate.** Sharpe and Sortino use a 0% hurdle. Over 2019–2023,
  rates went from ~2.4% to ~5%, so real risk-adjusted numbers are *worse* than
  those shown.
- **Pairs re-selection is fixed per fold, not rolling.** Within a fold, the
  pair is chosen once from the formation window. A production system would
  re-select continuously.
- **Five years is a short sample.** It contains one crash and one bear market.
  Five folds of walk-forward is enough to catch gross overfitting, not enough
  for statistical confidence.
- **A backtest is not live performance.** No slippage from latency, no partial
  fills, no exchange outages, no risk limits, no capital constraints.

---

## Repo layout

```
configs/     universe, strategy registry, backtest parameters (YAML)
data/cache/  committed per-ticker Parquet price cache
engine/      data_loader.py · backtest.py · run.py
strategies/  base.py · momentum.py · mean_reversion.py · pairs.py
metrics/     performance.py · validation.py
viz/         plots.py
scripts/     yahoo_browser_fetch.js · make_charts.py
tests/       64 tests
notebooks/results/   generated charts and results table
```

## Status

- [x] Phase 0 — Scope & setup
- [x] Phase 1 — Data layer
- [x] Phase 2 — Vectorized core engine
- [x] Phase 3 — Strategies
- [x] Phase 4 — Metrics & evaluation
- [x] Phase 5 — Visualization
- [ ] Phase 6 — Event-driven backtester *(stretch)*
- [ ] Phase 7 — C++ performance component *(stretch)*
- [x] Phase 8 — Polish & ship
