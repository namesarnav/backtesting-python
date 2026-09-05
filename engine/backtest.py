"""Phase 2: the vectorized core backtester.

Turns a table of "what should I own" into a table of "what happened to my
money", without ever using information that wasn't available at the time.

See PHASE2_GUIDE.md for the concepts. The short version of the design:

- **Everything is a whole-table operation.** There is no loop over dates
  anywhere in this file. Positions, returns, turnover and costs are each
  computed for all days and all tickers in one go, which is both faster and
  (once you're used to reading it) shorter than the equivalent loop.

- **Positions are lagged by `lag_days`.** This is the one line that keeps a
  vectorized backtest honest, and it has its own test. See `_lag_positions`.

- **Weights are normalized by *gross* exposure** (sum of absolute weights),
  not net. A long/short book of +0.5 / -0.5 nets to zero but is very much
  invested; dividing by the net sum would divide by zero. Gross normalization
  means "always 1x gross exposure, split according to the signal".

- **Costs are charged on changes in the position actually held**, i.e. after
  the lag, because that is the day the trade really happens.

Conventions chosen here (all defensible, all worth being able to justify):
  * turnover = sum of |change in weight|, counting both the buy and the sell
    sides, because a broker charges you on each.
  * the initial move from flat into a full portfolio IS charged. It's a real
    trade with a real cost, and over-charging is the safer bias in a backtest.
  * day 0 has no prior close, so it has no return and (after the lag) no
    position. Its portfolio return is 0.0 rather than NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Basis points -> plain fraction. 5bp = 5/10_000 = 0.0005.
BPS_PER_UNIT = 10_000.0

# Position changes smaller than this are floating-point noise, not trades.
TRADE_TOLERANCE = 1e-12

VALID_ALLOCATIONS = ("equal_weight", "vol_weighted")


@dataclass
class BacktestResult:
    """Everything the engine produces, bundled into one object.

    `returns` is the headline output; the rest exists so a result that looks
    wrong can be taken apart and inspected (and so Phase 4 can compute
    turnover-based metrics without recomputing them).
    """

    returns: pd.Series  # daily portfolio return, AFTER costs
    equity_curve: pd.Series  # (1 + returns).cumprod() -- what $1 became
    positions: pd.DataFrame  # weights actually held each day (lagged)
    trades: pd.DataFrame  # long-format log of every position change
    gross_returns: pd.Series = field(repr=False)  # before costs
    costs: pd.Series = field(repr=False)  # cost drag per day
    turnover: pd.Series = field(repr=False)  # sum |dw| per day


class VectorizedBacktester:
    """Runs a signal panel against a price panel and reports the outcome.

    Parameters
    ----------
    config : dict
        Typically `configs/backtest.yaml` loaded into a dict. Recognized keys:
        `allocation`, `lag_days`, `transaction_cost_bps`, `slippage_bps`,
        `vol_lookback`.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}

        self.allocation = config.get("allocation", "equal_weight")
        if self.allocation not in VALID_ALLOCATIONS:
            raise ValueError(
                f"unknown allocation {self.allocation!r}; expected one of {VALID_ALLOCATIONS}"
            )

        self.lag_days = int(config.get("lag_days", 1))
        if self.lag_days < 1:
            # Refusing 0 here on purpose. A lag of 0 lets a signal derived from
            # day t's closing price be traded on day t itself -- i.e. it trades
            # on a price that had not been printed yet when the decision was
            # made. That is look-ahead bias, and it is not a mode we offer.
            raise ValueError(
                f"lag_days must be >= 1 (got {self.lag_days}): a lag of 0 would let a "
                "signal computed from day t's close trade on day t, which is look-ahead bias"
            )

        self.transaction_cost_bps = float(config.get("transaction_cost_bps", 0.0))
        self.slippage_bps = float(config.get("slippage_bps", 0.0))
        # Both are charged the same way (a rate on notional traded), so collapse
        # them into one number once, here, rather than at every use site.
        self.cost_rate = (self.transaction_cost_bps + self.slippage_bps) / BPS_PER_UNIT

        self.vol_lookback = int(config.get("vol_lookback", 20))

    # -- validation -------------------------------------------------------

    @staticmethod
    def _validate(close: pd.DataFrame, signals: pd.DataFrame) -> None:
        """Reject mismatched inputs loudly.

        pandas would otherwise silently outer-join mismatched labels and fill
        the holes with NaN, turning a wiring mistake into quietly wrong
        numbers instead of an error. Catching it here is the difference
        between a five-second fix and a week of confusion.
        """
        if not isinstance(close, pd.DataFrame) or not isinstance(signals, pd.DataFrame):
            raise TypeError("close and signals must both be DataFrames")

        if not close.index.equals(signals.index):
            raise ValueError(
                "close and signals must share an identical date index "
                f"({len(close.index)} vs {len(signals.index)} rows)"
            )

        if not close.columns.equals(signals.columns):
            missing = sorted(set(close.columns) - set(signals.columns))
            extra = sorted(set(signals.columns) - set(close.columns))
            raise ValueError(
                "close and signals must share identical tickers in the same order; "
                f"missing from signals: {missing or 'none'}, "
                f"unexpected in signals: {extra or 'none'}"
            )

        if close.empty:
            raise ValueError("close panel is empty")

    # -- the pipeline -----------------------------------------------------

    @staticmethod
    def _asset_returns(close: pd.DataFrame) -> pd.DataFrame:
        """Daily simple return per ticker.

        Day 0 has no previous close, so `pct_change` leaves NaN there; it
        becomes 0.0, which is correct in the only sense that matters -- we
        hold nothing on day 0 anyway, so it contributes nothing either way.
        DataLoader guarantees no interior NaNs, so nothing else is being
        masked here.
        """
        return close.pct_change().fillna(0.0)

    def _target_weights(
        self, signals: pd.DataFrame, asset_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Turn raw signals into normalized target weights.

        Uses only information available up to and including each row's own
        date. The lag that makes those weights *tradeable* is applied
        separately, in `_lag_positions`.
        """
        raw = signals.astype(float)

        if self.allocation == "vol_weighted":
            # Size inversely to recent volatility, so one jumpy name doesn't
            # dominate the portfolio's risk. rolling().std() at row t uses
            # returns up to and including t -- same information cutoff as the
            # signal itself, so lagging the finished weights covers both.
            vol = asset_returns.rolling(
                self.vol_lookback, min_periods=self.vol_lookback
            ).std()
            raw = raw / vol
            # A ticker that hasn't moved at all gives vol == 0 -> inf. Drop
            # those rather than letting one name take the entire book.
            raw = raw.replace([np.inf, -np.inf], np.nan)

        # Normalize by GROSS exposure, not net: a +0.5/-0.5 long-short book
        # sums to zero but is fully invested. `where(gross > 0)` turns the
        # all-flat rows into NaN so the division doesn't blow up; fillna puts
        # them back to a legitimate 0.0 (in cash, holding nothing).
        gross = raw.abs().sum(axis=1)
        weights = raw.div(gross.where(gross > 0), axis=0)
        return weights.fillna(0.0)

    def _lag_positions(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Shift target weights forward so they're only tradeable later.

        *** THE LOOK-AHEAD GUARD. The most important line in this file. ***

        A signal on row t was computed from data through day t -- including
        day t's closing price, which does not exist until day t is over. So
        it cannot be acted on until day t+1. `shift(lag_days)` moves each
        row's weights forward in time, making

            position[t] = weight[t - lag_days]

        The first `lag_days` rows have nothing to inherit and become 0.0:
        before any signal existed we were flat, holding nothing.

        Delete this shift and `tests/test_backtest.py`'s look-ahead test
        should fail loudly. If it doesn't, the test isn't doing its job.
        """
        return weights.shift(self.lag_days).fillna(0.0)

    @staticmethod
    def _turnover(positions: pd.DataFrame) -> pd.Series:
        """How much of the book changed each day -- the base for costs.

        Absolute values, because selling costs money too: a sale is not a
        negative cost. Both sides are counted (swapping 0.2 from one name to
        another is 0.4 of turnover), because a broker bills you on each leg.
        The first row's `shift` NaN becomes 0.0, encoding "we started flat",
        which makes the initial buy-in show up as real, chargeable turnover.
        """
        previous = positions.shift(1).fillna(0.0)
        return (positions - previous).abs().sum(axis=1)

    @staticmethod
    def _trade_log(positions: pd.DataFrame) -> pd.DataFrame:
        """Long-format record of every position change.

        Only rows that actually changed -- logging a 476-ticker book over
        1,258 days of mostly-unchanged positions would bury the signal in
        noise.

        Select first, then build. The obvious implementation melts all three
        frames to long format and merges them, which materializes
        `dates x tickers` rows three times over and joins them before
        throwing nearly all of it away. That is invisible on a 40-ticker
        panel and dominant on a 476-ticker one: it cost ~180ms against ~10ms
        of actual backtest arithmetic, and it cost the same 180ms for the
        pairs strategy, whose entire trade log is 36 rows. Taking the
        non-zero coordinates up front makes the work proportional to the
        number of trades rather than to the size of the book.
        """
        previous = positions.shift(1).fillna(0.0)
        delta = positions - previous

        delta_values = delta.to_numpy()
        rows, cols = np.nonzero(np.abs(delta_values) > TRADE_TOLERANCE)

        # np.nonzero yields row-major order, i.e. already sorted by date and
        # then by column position; the panel's columns are sorted by ticker,
        # so this is the ["date", "ticker"] ordering the old sort produced.
        return pd.DataFrame(
            {
                "date": positions.index.to_numpy()[rows],
                "ticker": positions.columns.to_numpy()[cols],
                "prev_weight": previous.to_numpy()[rows, cols],
                "new_weight": positions.to_numpy()[rows, cols],
                "delta": delta_values[rows, cols],
            }
        )

    # -- public API -------------------------------------------------------

    def run(self, close: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        """Backtest `signals` against `close` prices.

        Parameters
        ----------
        close : DataFrame
            Dates x tickers of adjusted closing prices, e.g.
            `DataLoader().load_panel()["close"]`.
        signals : DataFrame
            Same shape as `close`. Values are desired exposure per ticker;
            they are normalized here, so their scale doesn't matter -- only
            their relative size and sign.
        """
        self._validate(close, signals)

        asset_returns = self._asset_returns(close)

        # Order matters: decide -> wait a day -> hold -> pay for the change.
        target_weights = self._target_weights(signals, asset_returns)
        positions = self._lag_positions(target_weights)

        # The whole portfolio calculation, for every day and ticker at once:
        # what we held, times what it did, summed across tickers (axis=1
        # collapses the columns, leaving one number per day).
        gross_returns = (positions * asset_returns).sum(axis=1)

        # Costs come off the position we actually held, i.e. post-lag, since
        # that is the day the trade genuinely takes place.
        turnover = self._turnover(positions)
        costs = turnover * self.cost_rate
        returns = gross_returns - costs

        # Returns compound: +10% then -10% leaves you at 0.99, not 1.00.
        equity_curve = (1.0 + returns).cumprod()

        return BacktestResult(
            returns=returns.rename("returns"),
            equity_curve=equity_curve.rename("equity_curve"),
            positions=positions,
            trades=self._trade_log(positions),
            gross_returns=gross_returns.rename("gross_returns"),
            costs=costs.rename("costs"),
            turnover=turnover.rename("turnover"),
        )
