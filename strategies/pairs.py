"""Pairs trading: trade the gap between two stocks that move together.

The idea. Some pairs of stocks are economically tied -- two refiners, two
railroads, two banks -- so while each wanders unpredictably, the *difference*
between them tends to be stable. When that difference stretches unusually
wide, you bet on it closing: buy the laggard, short the leader, and profit
when they converge, regardless of what the overall market does. That last
part is the appeal -- the market direction largely cancels out.

The statistical name for "these two wander together and their difference is
stable" is **cointegration**. Two price series can each be a random walk
(individually unpredictable) while a specific linear combination of them is
mean-reverting. `statsmodels`' `coint()` runs the Engle-Granger test: it
regresses one series on the other and asks whether the leftover residual is
stationary. A low p-value says "the gap between these two reliably comes
back".

The spread we trade is that regression's residual:

    spread = A - (alpha + beta * B)

`beta` is the **hedge ratio** -- how many units of B offset one unit of A.
If A is twice as volatile as B, you need about two units of B per unit of A
for the market exposure to cancel.

---

**The look-ahead trap this file exists to avoid.**

The obvious implementation is: test every pair over the whole 5-year history,
pick the most cointegrated one, then backtest it over that same 5 years. That
produces beautiful results and is completely worthless. You used the full
history to choose the pair, so you are trading on the knowledge that these
two *turned out* to stay together -- knowledge you could not have had at the
start. It is look-ahead bias wearing a lab coat, and it is called
**selection bias**.

It is a nastier bug than a missing `shift()`, because nothing in the
day-to-day arithmetic looks wrong. Every individual signal respects its
information cutoff. The leak is in the choice of *what to trade*.

The fix used here: a **formation window**. The first `lookback` rows are used
only to pick the pair and fit the hedge ratio -- no trading happens during
them. Everything after that is genuinely out of sample with respect to the
selection decision.

Documented simplification: the pair is chosen once, from that first window,
and held for the whole backtest. A production version would re-run selection
periodically (a rolling formation window), since a pair that cointegrated in
2019 may have decoupled by 2023. Phase 4's walk-forward validation is the
natural place to see whether that matters.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from strategies.base import Strategy


class PairsStrategy(Strategy):
    """Trade the mean-reverting spread of the most cointegrated pair found.

    Parameters
    ----------
    lookback : int
        Length of the formation window (used to select the pair and fit the
        hedge ratio) and of the rolling window for the spread's z-score.
    entry_z : float
        Open when the spread is this many standard deviations from its mean.
    exit_z : float
        Close when it comes back inside this. Must be < `entry_z`.
    pvalue_threshold : float
        Maximum cointegration p-value to accept a pair at all. If nothing
        clears this bar, the strategy holds nothing -- which is the correct
        answer, not a failure.
    n_candidates : int
        Cointegration tests aren't free and a 40-ticker universe has 780
        possible pairs. Prescreen by correlation (cheap) and only run the
        real test on the most promising `n_candidates`.
    """

    name = "pairs"

    def __init__(
        self,
        lookback: int = 252,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        pvalue_threshold: float = 0.05,
        n_candidates: int = 20,
    ):
        if lookback < 20:
            raise ValueError(f"lookback must be >= 20 for a meaningful test, got {lookback}")
        if entry_z <= exit_z:
            raise ValueError(f"entry_z ({entry_z}) must exceed exit_z ({exit_z})")

        self.lookback = int(lookback)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.pvalue_threshold = float(pvalue_threshold)
        self.n_candidates = int(n_candidates)

        # Populated by generate_signals, for inspection and testing.
        self.selected_pair: tuple[str, str] | None = None
        self.pvalue: float | None = None
        self.beta: float | None = None
        self.alpha: float | None = None

    # -- pair selection (formation window only) ---------------------------

    def _rank_candidates(self, formation: pd.DataFrame) -> list[tuple[str, str]]:
        """Cheapest-first prescreen: most correlated pairs are likeliest."""
        correlation = formation.pct_change().corr().abs()
        pairs = []
        for a, b in itertools.combinations(formation.columns, 2):
            value = correlation.at[a, b]
            if np.isfinite(value):
                pairs.append((value, (a, b)))
        pairs.sort(key=lambda item: item[0], reverse=True)
        return [pair for _, pair in pairs[: self.n_candidates]]

    def _select_pair(self, formation: pd.DataFrame):
        """Run the Engle-Granger test on the shortlist; keep the best pair."""
        best = (None, np.inf)
        for a, b in self._rank_candidates(formation):
            series_a, series_b = formation[a], formation[b]
            if series_a.std() == 0 or series_b.std() == 0:
                continue
            try:
                _, pvalue, _ = coint(series_a, series_b)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if pvalue < best[1]:
                best = ((a, b), pvalue)

        pair, pvalue = best
        if pair is None or pvalue > self.pvalue_threshold:
            return None, pvalue if pair is not None else None

        return pair, pvalue

    @staticmethod
    def _hedge_ratio(series_a: pd.Series, series_b: pd.Series) -> tuple[float, float]:
        """OLS of A on B over the formation window: A = alpha + beta*B.

        Same regression `coint` runs internally, so the spread we trade is
        the residual whose stationarity was actually tested.
        """
        design = np.column_stack([np.ones(len(series_b)), series_b.to_numpy()])
        (alpha, beta), *_ = np.linalg.lstsq(design, series_a.to_numpy(), rcond=None)
        return float(alpha), float(beta)

    # -- signal generation ------------------------------------------------

    def generate_signals(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        close = self._close_prices(price_panel)
        signals = self._blank_signals(close)

        if len(close) <= self.lookback:
            # Not enough history to even form a view. Hold nothing.
            return signals

        # ---- formation window: selection happens here and ONLY here ----
        formation = close.iloc[: self.lookback]
        pair, pvalue = self._select_pair(formation)
        self.pvalue = pvalue

        if pair is None:
            # Nothing cointegrated enough to trade. Sitting in cash is a
            # legitimate outcome, not an error.
            self.selected_pair = None
            return signals

        a, b = pair
        self.selected_pair = pair
        self.alpha, self.beta = self._hedge_ratio(formation[a], formation[b])

        # ---- trading: uses the fixed alpha/beta from formation ----
        spread = close[a] - (self.alpha + self.beta * close[b])

        # Trailing z-score of the spread -- backward-looking, like every
        # other rolling window in this project.
        mean = spread.rolling(self.lookback, min_periods=self.lookback).mean()
        std = spread.rolling(self.lookback, min_periods=self.lookback).std()
        z_score = ((spread - mean) / std.where(std > 0)).to_frame("spread")

        # invert=True: a spread far ABOVE its mean means A is rich relative
        # to B, so we short the spread (short A, long B) and wait for it to
        # close.
        position = self._apply_entry_exit(
            z_score.fillna(0.0), entry=self.entry_z, exit_=self.exit_z, invert=True
        )["spread"]
        position = position.where(z_score["spread"].notna(), 0.0)

        # Never trade inside the formation window: those prices chose the pair.
        position.iloc[: self.lookback] = 0.0

        # +1 means long the spread: long one unit of A, short beta units of B.
        signals[a] = position
        signals[b] = -position * self.beta
        return signals
