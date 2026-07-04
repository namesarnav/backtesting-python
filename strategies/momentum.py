"""Cross-sectional momentum: buy what has been going up.

The oldest and most studied anomaly in equities. The claim is that stocks
which outperformed their peers over the last few months tend to keep
outperforming over the next few -- Jegadeesh & Titman documented it in 1993
and it has been picked over ever since.

The rule here is deliberately plain:

  1. every day, measure each ticker's return over the trailing `lookback`
     days (126 trading days ~ 6 months by default)
  2. rank the tickers against each other *on that day*
  3. go long the top decile; optionally short the bottom decile

Two details worth noticing, because they are what make it a *portfolio*
strategy rather than a bet on one stock:

**It is cross-sectional, not absolute.** The comparison is stock-vs-stock on
the same day, never stock-vs-its-own-history. In a market where everything
fell 30%, this still holds the names that fell least. That makes it a bet on
*relative* strength, which is a very different (and more hedgeable) claim
than "this stock will go up".

**Ranking is done per row.** `rank(axis=1)` ranks across tickers within each
date. Ranking down a column instead would compare today's momentum against
last year's -- a subtle but complete change of meaning, and a classic bug.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class MomentumStrategy(Strategy):
    """Long the strongest recent performers, optionally short the weakest.

    Parameters
    ----------
    lookback : int
        Trading days of trailing return used to measure momentum. 126 is
        roughly six months.
    long_decile : float
        Fraction of the universe to hold long, e.g. 0.1 = top 10%.
    short_decile : float
        Fraction to short. Ignored when `long_only` is True.
    long_only : bool
        If True (the default in `configs/strategies.yaml`), never short.
    """

    name = "momentum"

    def __init__(
        self,
        lookback: int = 126,
        long_decile: float = 0.1,
        short_decile: float = 0.1,
        long_only: bool = True,
    ):
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        for label, value in (("long_decile", long_decile), ("short_decile", short_decile)):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{label} must be in (0, 1], got {value}")

        self.lookback = int(lookback)
        self.long_decile = float(long_decile)
        self.short_decile = float(short_decile)
        self.long_only = bool(long_only)

    def generate_signals(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        close = self._close_prices(price_panel)

        # Trailing return: close[t] / close[t - lookback] - 1. Strictly
        # backward-looking -- the newest price it touches is today's, which
        # is exactly the information cutoff the engine's 1-day lag assumes.
        trailing_return = close.pct_change(self.lookback)

        # Rank ACROSS TICKERS within each day (axis=1), as a percentile so
        # the thresholds don't depend on universe size. pct=True gives
        # (0, 1]; the strongest name on a given day scores 1.0.
        rank = trailing_return.rank(axis=1, pct=True, na_option="keep")

        signals = self._blank_signals(close)
        signals = signals.mask(rank > 1.0 - self.long_decile, 1.0)
        if not self.long_only:
            signals = signals.mask(rank <= self.short_decile, -1.0)

        # The first `lookback` rows have no trailing return, so rank is NaN
        # and every comparison above is False -- those days are already flat.
        # This just guarantees it rather than relying on that behaviour.
        return signals.where(rank.notna(), 0.0)
