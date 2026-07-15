"""Performance metrics: turning a return series into numbers you can compare.

Every function here takes a daily return series (what
`VectorizedBacktester.run().returns` produces) and returns one number.

Two conventions used throughout, both worth being able to defend:

**Annualization uses 252 trading days**, not 365 calendar days. Markets are
shut on weekends and holidays, and a return series only has entries for days
that traded, so 252 is the count that matches the data.

**Sharpe and Sortino are computed arithmetically** -- mean daily excess
return divided by daily standard deviation, scaled by sqrt(252) -- which is
the industry-standard form. `annualized_return` is separately *geometric*
(it compounds), because that is what actually happened to your money. The two
differ, and mixing them up produces a Sharpe that quietly disagrees with
everyone else's. They are kept distinct on purpose.

**On infinities.** The spec's Phase 4 checkpoint says to be suspicious of
infinite Sharpe ratios. They arise from dividing by a zero standard
deviation, which happens whenever a strategy never traded over the window
being measured -- a real and common situation in walk-forward, where a fold
can easily contain no positions at all. Every ratio here returns NaN rather
than inf in that case: "undefined" is the honest answer, and unlike inf it
won't poison a mean or silently top a leaderboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaNs and guarantee a float Series to work with."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    return returns.astype(float).dropna()


def total_return(returns: pd.Series) -> float:
    """Compounded return over the whole period. 0.5 means the book grew 50%."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    return float((1.0 + r).prod() - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric average annual growth rate (CAGR).

    Geometric, not arithmetic: +50% then -50% averages to 0% arithmetically
    but leaves you down 25%, and only the geometric figure says so.
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    growth = (1.0 + r).prod()
    if growth <= 0:
        # A strategy that lost everything has no meaningful growth *rate*.
        return -1.0
    return float(growth ** (periods_per_year / len(r)) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Standard deviation of returns, scaled to a yearly figure.

    Volatility scales with the square root of time, hence sqrt(252) rather
    than 252 -- a consequence of variance being additive over independent
    periods while standard deviation is its square root.
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Return earned per unit of risk taken.

    The single most quoted number in the industry: how much excess return you
    got for the volatility you endured. Roughly, below 1 is unremarkable,
    above 2 is very good, and above 3 in a backtest is a reason to go looking
    for look-ahead bias rather than to celebrate.

    `risk_free_rate` is an annual figure (0.05 = 5%), converted to a daily
    hurdle internally.
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan

    excess = r - risk_free_rate / periods_per_year
    sigma = excess.std(ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        # Never traded, or a constant return: no risk taken means the ratio
        # is undefined, not infinite.
        return np.nan
    return float(excess.mean() / sigma * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Sharpe, but only penalizing downside volatility.

    Sharpe treats a violent move up as identically "risky" to a violent move
    down, which does not match how anyone actually experiences a portfolio.
    Sortino divides by *downside deviation* -- the volatility of losses alone
    -- so a strategy is not punished for its good days.

    Downside deviation uses the full-length root-mean-square of the negative
    part (dividing by n, not by the count of losing days). Dividing by the
    number of losses instead would flatter a strategy that rarely loses but
    loses badly.
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan

    excess = r - risk_free_rate / periods_per_year
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt((downside**2).sum() / len(excess)))
    if downside_deviation == 0:
        # No losing days at all. Undefined rather than infinite -- and worth
        # a raised eyebrow if a real strategy reports it.
        return np.nan
    return float(excess.mean() / downside_deviation * np.sqrt(periods_per_year))


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative growth of one unit of capital."""
    return (1.0 + _clean(returns)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough fall, as a negative fraction.

    The number that decides whether a strategy is survivable in practice: a
    great long-run return is irrelevant if the path there involves a 70%
    decline nobody would sit through.
    """
    r = _clean(returns)
    if r.empty:
        return np.nan
    curve = (1.0 + r).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annual return divided by the depth of the worst drawdown.

    Answers "how much did I earn per unit of worst-case pain?", which is
    often a more visceral risk measure than volatility.
    """
    drawdown = max_drawdown(returns)
    if drawdown is None or not np.isfinite(drawdown) or drawdown == 0:
        return np.nan
    return float(annualized_return(returns, periods_per_year) / abs(drawdown))


def win_rate(returns: pd.Series) -> float:
    """Fraction of days with a positive return.

    Deliberately says nothing about *size*. A strategy can win 90% of days
    and still lose money if the 10% are catastrophic, so this is only ever
    read alongside the return figures.
    """
    r = _clean(returns)
    # Days flat at exactly zero are excluded: for a strategy that sits in
    # cash most of the time, counting those as losses would be misleading.
    traded = r[r != 0]
    if traded.empty:
        return np.nan
    return float((traded > 0).mean())


def average_turnover(turnover: pd.Series) -> float:
    """Mean fraction of the book traded per day.

    0.2 means roughly a fifth of the portfolio changes hands daily -- about
    50x a year, which at 7bp a turn is ~3.5% of capital lost annually to
    costs before the strategy is right or wrong about anything.
    """
    t = _clean(turnover)
    return np.nan if t.empty else float(t.mean())


def summarize(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """All of the above in one dict, for building results tables."""
    summary = {
        "total_return": total_return(returns),
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "n_periods": int(len(_clean(returns))),
    }
    if turnover is not None:
        summary["avg_turnover"] = average_turnover(turnover)
    return summary
