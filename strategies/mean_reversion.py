"""Mean reversion: fade stocks that have stretched away from their average.

The opposite bet to momentum, over a much shorter horizon. Momentum says
"the trend continues"; mean reversion says "that move was an overreaction and
will partly undo itself". Both are real effects -- they operate on different
timescales, which is why a portfolio can run them side by side.

The measure is a **z-score**: how many standard deviations today's price sits
from its own recent average.

    z = (price - rolling_mean) / rolling_std

Worked example. A stock's 20-day average is $100 and its 20-day standard
deviation is $2:

    price $104  ->  z = (104 - 100) / 2 = +2.0   two sigma rich  -> SHORT
    price $100  ->  z =   0.0                    at its average  -> flat
    price  $97  ->  z = (97 - 100) / 2 = -1.5    stretched cheap -> LONG

Dividing by the standard deviation is what makes the number comparable
across tickers: a $4 move in a placid utility and a $4 move in a volatile
tech name are not the same event, but "+2 sigma" means the same thing in
both. This is the same normalize-by-volatility idea the engine's
`vol_weighted` allocation uses, applied to signal generation instead of
position sizing.

**Note the sign.** High z-score means *short*, not long -- the whole premise
is that the move reverses. That is `invert=True` in `_apply_entry_exit`, and
getting it backwards turns this into a (bad, slow) momentum strategy. The
unit tests pin the direction down for exactly that reason.

**Entry and exit thresholds differ on purpose.** Enter at |z| >= 1.0, exit
only once |z| <= 0.25. A single threshold would have positions flickering on
and off every time the z-score jittered across the line, and since the engine
charges cost on every position change, that churn is expensive. The gap
between the two thresholds is a deliberate brake on turnover.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """Long oversold names, short overbought ones, on a rolling z-score.

    Parameters
    ----------
    window : int
        Trading days in the rolling mean/std used to define "normal".
    entry_z : float
        Open a position once |z| reaches this.
    exit_z : float
        Close it once |z| falls back to this. Must be < `entry_z`.
    """

    name = "mean_reversion"

    def __init__(self, window: int = 20, entry_z: float = 1.0, exit_z: float = 0.25):
        if window < 2:
            raise ValueError(f"window must be >= 2 to have a standard deviation, got {window}")
        if entry_z <= exit_z:
            raise ValueError(
                f"entry_z ({entry_z}) must exceed exit_z ({exit_z}), or positions "
                "would close the moment they open"
            )

        self.window = int(window)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)

    def generate_signals(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        close = self._close_prices(price_panel)

        # `rolling` looks backwards by default: the window ending at row t
        # covers t-window+1 .. t. min_periods keeps the first `window` rows
        # NaN rather than computing a mean from two observations.
        rolling_mean = close.rolling(self.window, min_periods=self.window).mean()
        rolling_std = close.rolling(self.window, min_periods=self.window).std()

        # A flat-lined stock has std 0; leave it NaN rather than dividing by
        # zero and declaring an infinitely extreme move.
        z_score = (close - rolling_mean) / rolling_std.where(rolling_std > 0)

        # invert=True: a HIGH z-score is overbought, which is a short.
        signals = self._apply_entry_exit(
            z_score.fillna(0.0), entry=self.entry_z, exit_=self.exit_z, invert=True
        )

        # Stay flat until the rolling window is actually full.
        return signals.where(z_score.notna(), 0.0)
