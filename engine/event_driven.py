"""Phase 6: the event-driven backtester.

The Phase 2 engine answers "what would this strategy have returned?" in a
handful of whole-table operations. This one answers the same question by
*simulating*: one bar at a time, holding explicit state -- a cash balance and
a share count per ticker -- and placing orders that get filled.

Why build the same thing twice
------------------------------
The vectorized engine works in **weights**. It never asks how many shares
that is, where the cash came from, or whether the order could actually be
filled. Those questions don't exist in its universe, so it can never get them
wrong -- and can never warn you about them either.

The event-driven engine works in **shares and cash**, which makes a whole
class of real-world frictions expressible:

  * a position drifts as prices move, so holding a *constant weight* means
    trading every single day
  * you can only buy whole shares (optional here, on by default in reality)
  * costs come out of the cash balance, not out of an abstract return
  * an order can be skipped if it's too small to be worth the commission

Real trading infrastructure is event-driven because the live system *is* an
event loop -- the same code path that backtests must be the one that trades,
or the backtest is testing something the production system doesn't do.

The relationship to the vectorized engine
-----------------------------------------
Both engines share the allocation step. `signal panel -> normalized target
weights` is literally `VectorizedBacktester._target_weights`, called from
here. That is deliberate: it means the two engines cannot disagree about
*what to hold*, so every difference in the results is attributable to *how it
gets held*, which is the only interesting comparison.

With fractional shares, no rebalance threshold and fills at the close, the
two engines produce **identical gross returns** (to floating-point noise).
See `tests/test_event_driven.py`. The net returns differ, always in the same
direction, for one honest reason:

    The vectorized engine computes turnover from *target* weights:
    |w[t] - w[t-1]|. But the weights it actually held drifted overnight as
    prices moved. Correcting that drift is a real trade that a real broker
    really bills you for, and the vectorized engine cannot see it. This
    engine can, so it charges more.

That is not a bug in either one. It is the vectorized engine's cost model
being an approximation, and the event-driven engine measuring the thing the
approximation approximates.

Two smaller differences, both real and both worth knowing about:

  * **Cost timing.** A trade decided from bar t's close is filled at bar t's
    close, so the cash leaves at bar t. The vectorized engine charges it
    against bar t+1, the day the position becomes effective. A one-bar shift
    in the cost series; immaterial to the total, visible day by day.

  * **Sizing on pre-commission equity.** Orders are sized off the equity we
    have *before* paying for them, because the commission is not known until
    the order exists. After paying it we are therefore holding a fraction of
    a basis point more than 100% gross. This is what a live system does, and
    it means gross returns match the vectorized engine *exactly* only when
    costs are zero; with costs on they agree to about 1e-5 relative.

Phase 7 note
------------
`_simulate` is the hot loop -- genuinely serial Python, one iteration per bar,
which is exactly the shape that a C++ port speeds up (and exactly the shape
that NumPy cannot). It takes and returns plain arrays for that reason: the
seam is already cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.backtest import VectorizedBacktester

# An order counts as a real fill when the dollars it moves exceed this
# fraction of the portfolio. Deliberately *not* an absolute tolerance on the
# share count, which the vectorized engine can get away with (it works in
# weights, which are already scale-free) but this engine cannot: 1e-12 shares
# of a $1 stock and of a $1000 stock are not the same event.
#
# It also has to survive float dust. Hold one asset at 100% and the target
# share count is exactly what you already own, so every bar's delta is pure
# rounding noise sitting near zero. Measured in shares that noise lands right
# on an absolute 1e-12 threshold and tips either way depending on summation
# order; measured as a fraction of equity it is ~1e-15, three orders of
# magnitude clear of the line. That is the difference between a criterion
# that is stable across implementations and one that is not.
FILL_TOLERANCE = 1e-12


@dataclass
class EventDrivenResult:
    """Same headline fields as `BacktestResult`, plus the simulation state.

    The shared field names are load-bearing: `metrics.summarize` and
    `viz.plots` take either result object without knowing which engine
    produced it.
    """

    returns: pd.Series  # daily portfolio return, AFTER costs
    equity_curve: pd.Series  # normalized to start at 1.0, like Phase 2
    positions: pd.DataFrame  # weights actually held, post-trade
    trades: pd.DataFrame  # one row per fill: shares, price, notional, cost
    gross_returns: pd.Series = field(repr=False)  # before costs
    costs: pd.Series = field(repr=False)  # cost drag per day, as a fraction
    turnover: pd.Series = field(repr=False)  # traded notional / equity
    cash: pd.Series = field(repr=False)  # cash balance in dollars
    holdings: pd.DataFrame = field(repr=False)  # share count per ticker
    equity: pd.Series = field(repr=False)  # portfolio value in dollars


# The Phase 7 C++ port of the bar loop. Optional on purpose: a clone without
# a compiler must still run the whole pipeline, so a missing extension is a
# fallback, not an error. Build it with `python setup.py build_ext --inplace`.
try:
    from engine import _fastloop  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on whether the ext is built
    _fastloop = None

VALID_BACKENDS = ("auto", "python", "cpp")


def available_backends() -> tuple[str, ...]:
    """Which loop implementations this install can actually run."""
    return ("python", "cpp") if _fastloop is not None else ("python",)


def _resolve_backend(backend: str) -> str:
    """Turn a requested backend into the one that will actually be used.

    `auto` prefers C++ and silently falls back; `cpp` is a demand, and fails
    loudly if the extension is missing rather than quietly benchmarking the
    Python loop and reporting it as the C++ number.
    """
    if backend not in VALID_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {VALID_BACKENDS}")
    if backend == "cpp" and _fastloop is None:
        raise RuntimeError(
            "backend='cpp' requested but engine._fastloop is not built. "
            "Run: python setup.py build_ext --inplace"
        )
    if backend == "auto":
        return "cpp" if _fastloop is not None else "python"
    return backend


def _simulate(*args, backend: str = "auto", **kwargs) -> dict:
    """Run the bar loop through the requested backend.

    Both implementations take the same arguments and return the same dict of
    arrays, which is what makes `tests/test_fastloop.py` able to assert they
    agree. See `cpp/event_loop.cpp` for the C++ side.
    """
    if _resolve_backend(backend) == "cpp":
        return _fastloop.simulate(*args, **kwargs)
    return _simulate_python(*args, **kwargs)


def _simulate_python(
    prices: np.ndarray,
    target_weights: np.ndarray,
    *,
    initial_capital: float,
    cost_rate: float,
    lag_days: int,
    fractional_shares: bool,
    rebalance_threshold: float,
) -> dict[str, np.ndarray]:
    """The bar-by-bar loop. Plain arrays in, plain arrays out.

    Each bar does four things, in this order, and the order is the whole
    point:

      1. **Mark to market.** Value what we already hold at today's price.
         This is where yesterday's positions earn (or lose) today's return.
      2. **Decide.** Look up the target weights (see the lag note below).
      3. **Order.** Convert target weights into target *share counts* using
         the freshly marked equity, and diff against what we hold.
      4. **Fill.** Trade at today's close, move the cash, pay the costs.

    Steps 1 and 4 both touch today's price, and that is not double-counting:
    we earn the return on the *old* position, then re-establish the position
    at the new price. A trade placed today cannot earn today's move.

    Where the lag actually lives -- read this before changing the indexing
    ---------------------------------------------------------------------
    Filling at the close of bar t means the position we establish today is
    exposed to bar t+1's move and nothing earlier. **The simulation is
    therefore already one bar lagged, structurally**, without any shift: a
    signal computed from today's close cannot touch today's return, because
    today's return was earned by yesterday's holdings.

    That is the exact same convention as Phase 2, where `positions[t] =
    w[t-1]` earns `ret[t] = P[t]/P[t-1] - 1` -- a position established at
    price `P[t-1]` using a signal computed from `P[t-1]`.

    So `lag_days=1` maps to *zero* extra delay here, and the general rule is:

        target for bar t = target_weights[t - (lag_days - 1)]

    Indexing `target_weights[t - lag_days]` instead would double-lag the
    book: correct-looking, no test would obviously break, and the two
    engines would quietly disagree by one bar forever.
    """
    execution_delay = lag_days - 1
    n_bars, n_assets = prices.shape

    shares = np.zeros(n_assets)
    cash = float(initial_capital)

    equity = np.empty(n_bars)
    gross_returns = np.zeros(n_bars)
    net_returns = np.zeros(n_bars)
    costs = np.zeros(n_bars)
    traded_notional = np.zeros(n_bars)
    cash_history = np.empty(n_bars)
    holdings = np.empty((n_bars, n_assets))
    weights_established = np.zeros((n_bars, n_assets))
    # Fills are accumulated as parallel columns rather than a list of
    # tuples: it is the shape the C++ backend can hand back cheaply, and it
    # keeps `_fill_log` identical for both.
    fill_bar: list[int] = []
    fill_asset: list[int] = []
    fill_shares: list[float] = []
    fill_price: list[float] = []
    fill_notional: list[float] = []
    fill_cost: list[float] = []

    previous_equity = float(initial_capital)

    for t in range(n_bars):
        price = prices[t]

        # 1. Mark to market: what we're worth before doing anything today.
        equity_pre = cash + shares @ price

        # 2. Decide. Before enough bars have elapsed there is no signal to
        #    act on, so we sit flat -- holding nothing, like Phase 2's
        #    `shift(...).fillna(0.0)`. See the docstring for why the offset
        #    is `lag_days - 1` and not `lag_days`.
        target = (
            target_weights[t - execution_delay]
            if t >= execution_delay
            else np.zeros(n_assets)
        )

        # 3. Order. Target dollar exposure -> target share count.
        target_shares = (target * equity_pre) / price
        if not fractional_shares:
            # Truncate toward zero, never away from it: rounding up would
            # quietly lever the book past 100% gross.
            target_shares = np.trunc(target_shares)

        delta = target_shares - shares
        notional = np.abs(delta) * price

        if rebalance_threshold > 0.0:
            # Skip orders too small to be worth the commission. The position
            # stays where it is, so its weight is left to drift.
            too_small = notional < rebalance_threshold * equity_pre
            delta = np.where(too_small, 0.0, delta)
            target_shares = shares + delta
            notional = np.abs(delta) * price

        # 4. Fill at today's close.
        bar_cost = float(notional.sum()) * cost_rate
        cash -= float(delta @ price) + bar_cost
        shares = target_shares

        equity_post = cash + shares @ price  # == equity_pre - bar_cost

        gross_returns[t] = equity_pre / previous_equity - 1.0
        net_returns[t] = equity_post / previous_equity - 1.0
        costs[t] = bar_cost / previous_equity
        traded_notional[t] = notional.sum() / previous_equity
        equity[t] = equity_post
        cash_history[t] = cash
        holdings[t] = shares
        # The weights we just established. They are exposed to bar t+1,
        # not bar t, so `run` shifts this by one before reporting it as
        # `positions` -- which is what makes it line up with Phase 2.
        weights_established[t] = shares * price / equity_post

        for i in np.nonzero(notional > FILL_TOLERANCE * equity_pre)[0]:
            fill_bar.append(t)
            fill_asset.append(int(i))
            fill_shares.append(float(delta[i]))
            fill_price.append(float(price[i]))
            fill_notional.append(float(delta[i] * price[i]))
            fill_cost.append(float(notional[i] * cost_rate))

        previous_equity = equity_post

    return {
        "equity": equity,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "costs": costs,
        "turnover": traded_notional,
        "cash": cash_history,
        "holdings": holdings,
        "weights_established": weights_established,
        "fill_bar": np.asarray(fill_bar, dtype=np.int64),
        "fill_asset": np.asarray(fill_asset, dtype=np.int64),
        "fill_shares": np.asarray(fill_shares, dtype=float),
        "fill_price": np.asarray(fill_price, dtype=float),
        "fill_notional": np.asarray(fill_notional, dtype=float),
        "fill_cost": np.asarray(fill_cost, dtype=float),
    }


class EventDrivenBacktester:
    """Bar-by-bar portfolio simulation with explicit cash and share state.

    Parameters
    ----------
    config : dict
        Everything `VectorizedBacktester` accepts (`allocation`, `lag_days`,
        `transaction_cost_bps`, `slippage_bps`, `vol_lookback`), plus:

        initial_capital : float, default 1_000_000
            Starting cash. Only matters when `fractional_shares` is False --
            with fractional shares the result is scale-invariant.
        fractional_shares : bool, default True
            False forces whole-share orders, which is what a real broker
            allows and which introduces tracking error against the
            vectorized engine (a $40 stock cannot express a 0.3% weight in a
            $10k account).
        rebalance_threshold : float, default 0.0
            Skip any order smaller than this fraction of equity. A no-trade
            band: cuts turnover and costs at the price of letting weights
            drift away from target.
        backend : {'auto', 'python', 'cpp'}, default 'auto'
            Which implementation of the bar loop to run. Results are the
            same either way (see `tests/test_fastloop.py`); only the speed
            differs. 'auto' uses C++ when the extension is built.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}

        # Allocation and validation are delegated, not duplicated: both
        # engines must agree on what to hold, or comparing them is
        # meaningless. This also inherits the lag_days >= 1 guard.
        self._allocator = VectorizedBacktester(config)

        self.lag_days = self._allocator.lag_days
        self.cost_rate = self._allocator.cost_rate

        self.initial_capital = float(config.get("initial_capital", 1_000_000.0))
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive (got {self.initial_capital})")

        self.fractional_shares = bool(config.get("fractional_shares", True))

        self.backend = str(config.get("backend", "auto"))
        _resolve_backend(self.backend)  # fail at construction, not mid-run

        self.rebalance_threshold = float(config.get("rebalance_threshold", 0.0))
        if self.rebalance_threshold < 0:
            raise ValueError(
                f"rebalance_threshold must be >= 0 (got {self.rebalance_threshold})"
            )

    def run(self, close: pd.DataFrame, signals: pd.DataFrame) -> EventDrivenResult:
        """Simulate `signals` against `close` prices, bar by bar.

        Takes and returns the same shapes as `VectorizedBacktester.run`, so
        the two are drop-in swappable in `engine/run.py`, the metrics layer
        and the charts.
        """
        self._allocator._validate(close, signals)

        if (close <= 0).to_numpy().any():
            # Share counts are notional/price. A zero or negative price is
            # either bad data or a corporate action the adjustment missed;
            # either way, dividing by it produces nonsense rather than an
            # error, so refuse it here.
            raise ValueError("close prices must all be positive to size positions in shares")

        asset_returns = self._allocator._asset_returns(close)
        target_weights = self._allocator._target_weights(signals, asset_returns)

        state = _simulate(
            close.to_numpy(dtype=float),
            target_weights.to_numpy(dtype=float),
            initial_capital=self.initial_capital,
            cost_rate=self.cost_rate,
            lag_days=self.lag_days,
            fractional_shares=self.fractional_shares,
            rebalance_threshold=self.rebalance_threshold,
            backend=self.backend,
        )

        index, tickers = close.index, close.columns

        def _series(key: str, name: str) -> pd.Series:
            return pd.Series(state[key], index=index, name=name)

        returns = _series("net_returns", "returns")

        return EventDrivenResult(
            returns=returns,
            # Normalized to 1.0 so it plots against the vectorized engine and
            # the benchmark on one axis, rather than in dollars.
            equity_curve=(1.0 + returns).cumprod().rename("equity_curve"),
            # Shifted by one bar: what was established at the close of t-1
            # is what we were actually exposed to during bar t. This makes
            # `positions` directly comparable to the vectorized engine's.
            positions=pd.DataFrame(
                state["weights_established"], index=index, columns=tickers
            ).shift(1).fillna(0.0),
            trades=self._fill_log(state, index, tickers),
            gross_returns=_series("gross_returns", "gross_returns"),
            costs=_series("costs", "costs"),
            turnover=_series("turnover", "turnover"),
            cash=_series("cash", "cash"),
            holdings=pd.DataFrame(state["holdings"], index=index, columns=tickers),
            equity=_series("equity", "equity"),
        )

    @staticmethod
    def _fill_log(state: dict, index: pd.Index, tickers: pd.Index) -> pd.DataFrame:
        """Long-format record of every fill.

        Richer than the vectorized engine's trade log, which can only report
        a change in weight: this knows the share count, the fill price, the
        dollars that moved and the commission paid on them.

        Takes the raw column arrays from either backend, so the C++ loop and
        the Python loop produce the same DataFrame through the same code.
        """
        columns = ["date", "ticker", "shares", "price", "notional", "cost"]
        if state["fill_bar"].size == 0:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            {
                "date": index[state["fill_bar"]],
                "ticker": tickers[state["fill_asset"]],
                "shares": state["fill_shares"],
                "price": state["fill_price"],
                "notional": state["fill_notional"],
                "cost": state["fill_cost"],
            }
        )


def reconcile(
    close: pd.DataFrame, signals: pd.DataFrame, config: dict | None = None
) -> dict:
    """Run both engines on one strategy and quantify the difference.

    This is the Phase 6 checkpoint in function form: it proves the engines
    agree where they must (gross returns) and reports by how much they
    differ where they legitimately can (costs, and therefore net returns).

    Returns a dict of scalars, all as plain fractions:
        max_gross_diff   -- largest single-day gross return gap. With
                            costs off this is floating-point noise (~1e-15)
                            and anything bigger is a bug: it means the
                            engines disagree about what to hold. With costs
                            on, expect ~1e-5, from sizing orders on
                            pre-commission equity (see the module docstring).
        vectorized_*     -- total return / avg turnover from Phase 2
        event_driven_*   -- the same from this engine
        cost_drag_diff   -- how much more the event-driven engine charged,
                            annualized-equivalent as a total-return gap
    """
    config = config or {}
    vectorized = VectorizedBacktester(config).run(close, signals)
    event_driven = EventDrivenBacktester(config).run(close, signals)

    gross_diff = (vectorized.gross_returns - event_driven.gross_returns).abs()

    return {
        "max_gross_diff": float(gross_diff.max()),
        "vectorized_total_return": float(vectorized.equity_curve.iloc[-1] - 1.0),
        "event_driven_total_return": float(event_driven.equity_curve.iloc[-1] - 1.0),
        "vectorized_avg_turnover": float(vectorized.turnover.mean()),
        "event_driven_avg_turnover": float(event_driven.turnover.mean()),
        "vectorized_total_cost": float(vectorized.costs.sum()),
        "event_driven_total_cost": float(event_driven.costs.sum()),
        "cost_drag_diff": float(event_driven.costs.sum() - vectorized.costs.sum()),
        "n_fills": int(len(event_driven.trades)),
    }
