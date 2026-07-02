"""The strategy contract every trading rule in this package implements.

One interface, `generate_signals(price_panel) -> signal_panel`, so the engine
never needs to know which strategy it is running. Momentum, mean reversion
and pairs trading are wildly different ideas, but they all answer the same
question -- "given prices, how much of each ticker should I want to hold?" --
and that is the only question `VectorizedBacktester` asks.

If a new idea does not fit this interface, the interface is the thing to
reconsider; adding an `if isinstance(strategy, ...)` branch inside the engine
is how a clean design rots.

**The look-ahead contract.** A signal on row `t` may use data up to and
including row `t`, and nothing after it. The engine then lags positions by a
day before trading them (see `engine/backtest.py::_lag_positions`), which is
what makes row `t`'s signal tradeable on `t+1`.

This means strategies must only ever look *backwards*:
  * `rolling(n)`, `pct_change(n)`, `shift(+n)`  -- fine, all backward-looking
  * `shift(-n)`, `rolling(n, center=True)`      -- look-ahead, never use these
  * fitting anything on the full history and then trading over that same
    history -- also look-ahead, and much easier to do by accident. See
    `strategies/pairs.py` for how that one is handled here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for every trading rule.

    Subclasses implement `generate_signals`. Values in the returned panel are
    *desired exposure*, not final weights -- the engine normalizes them by
    gross exposure, so only their relative size and sign matter. Returning
    +1/-1/0 is the common case.
    """

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """Return a (dates x tickers) panel of desired exposures.

        Must be the same shape as `price_panel["close"]`, contain no NaNs,
        and use no information from the future.
        """

    # -- shared helpers ---------------------------------------------------

    @staticmethod
    def _close_prices(price_panel: pd.DataFrame) -> pd.DataFrame:
        """Pull closing prices out of whatever panel shape was handed in.

        Accepts either the full `(field, ticker)` panel from `DataLoader` or
        a plain `(date x ticker)` frame of closes, so a strategy can be
        driven straight from the data layer or from a hand-built test frame
        without the caller having to care.
        """
        if isinstance(price_panel.columns, pd.MultiIndex):
            fields = price_panel.columns.get_level_values(0)
            if "close" not in fields:
                raise ValueError(
                    f"price panel has no 'close' field; got fields {sorted(set(fields))}"
                )
            return price_panel["close"]
        return price_panel

    @staticmethod
    def _blank_signals(close: pd.DataFrame) -> pd.DataFrame:
        """An all-zero signal panel shaped like `close` -- i.e. hold nothing."""
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

    @staticmethod
    def _apply_entry_exit(
        score: pd.DataFrame, entry: float, exit_: float, invert: bool = False
    ) -> pd.DataFrame:
        """Turn a z-score-like signal into held positions with hysteresis.

        Entry/exit thresholds mean a position, once opened, is *held* until
        the score decays back inside the exit band -- it isn't reopened from
        scratch every day. That needs memory of the previous day, which
        sounds like it needs a loop over dates. It doesn't:

          1. mark the days where something definite happens -- an entry
             (score beyond `entry`) or an exit (score back inside `exit_`)
          2. leave every other day as NaN, meaning "no news, carry on"
          3. `ffill()` propagates the last definite decision forward

        The result is a stateful-looking rule computed as whole-array
        operations. Days before the first entry stay NaN and become 0.0.

        `invert=False` means a high score is bullish (momentum-style).
        `invert=True` means a high score is bearish (mean-reversion-style:
        far above the mean is a short, not a buy).
        """
        if entry <= exit_:
            raise ValueError(
                f"entry threshold ({entry}) must be greater than exit threshold ({exit_}); "
                "otherwise a position would exit the instant it opened"
            )

        long_side, short_side = (-1.0, 1.0) if invert else (1.0, -1.0)

        target = pd.DataFrame(float("nan"), index=score.index, columns=score.columns)
        target = target.mask(score >= entry, short_side)
        target = target.mask(score <= -entry, long_side)
        # Exit band is checked last so it wins on any overlap; the constructor
        # guard above means there shouldn't be one.
        target = target.mask(score.abs() <= exit_, 0.0)

        return target.ffill().fillna(0.0)
