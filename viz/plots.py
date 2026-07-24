"""Charts: equity curves with drawdown, rolling Sharpe, and a comparison dashboard.

Saves PNGs to `notebooks/results/` for embedding in the README.

Design notes, since chart choices are as reviewable as code:

**Palette.** Three categorical hues (blue / orange / aqua) for the three
strategies, validated for colour-vision deficiency -- worst all-pairs CVD
separation dE 9.2, comfortably over the 8.0 target, and normal-vision dE 24.0
against a 15 floor. The aqua sits at 2.74:1 against the surface, under the 3:1
bar, so every series is **direct-labelled at its right edge**: identity never
rests on hue alone. The benchmark is deliberately *not* given a categorical
hue -- it is drawn as a grey dashed reference line, because it is the thing
being compared against rather than a fourth peer.

**Drawdown gets its own panel** rather than being shaded onto the equity
curve. Both are time series but they live on different scales (growth
multiple vs. percentage below peak), and putting two scales on one axis is
the most common chart mistake there is. Two stacked panels sharing an x-axis
says the same thing without lying about either.

**Equity is drawn on a log scale.** On a linear axis a move from 1.0 to 1.1
looks tiny while 3.0 to 3.3 looks huge, though both are +10%. Log scale makes
equal *percentage* moves equal distances, which is what you actually want to
compare between strategies.

**Known market events are shaded** -- the 2020 COVID crash and the 2022
selloff. That is partly context for the reader and partly a correctness check
in visual form: real equity curves must dive during those windows, and if
they don't, the data or the engine is wrong.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display in Docker/CI -- render straight to file.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics.performance import TRADING_DAYS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "results"

# Validated categorical slots (see module docstring).
SERIES_COLORS = {
    "momentum": "#2a78d6",        # blue
    "mean_reversion": "#eb6834",  # orange
    "pairs": "#1baf7a",           # aqua
}
BENCHMARK_COLOR = "#52514e"       # neutral ink -- a reference, not a series
FALLBACK_COLORS = ["#eda100", "#e87ba4", "#4a3aa7"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
DRAWDOWN_FILL = "#d03b3b"  # status:critical -- a drawdown is a loss, not a category

# Windows a real US equity curve must visibly react to. Used as annotation and
# as a visual correctness check.
MARKET_EVENTS = [
    ("2020-02-19", "2020-03-23", "COVID crash"),
    ("2022-01-03", "2022-10-14", "2022 selloff"),
]


def _style_axis(ax, ylabel: str | None = None) -> None:
    """Recessive grid and axes so the data carries the ink."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)


def _shade_events(ax, index: pd.DatetimeIndex, label: bool = True) -> None:
    """Grey bands over well-known market dislocations."""
    for start, end, name in MARKET_EVENTS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start < index[0] or end > index[-1]:
            continue
        ax.axvspan(start, end, color=GRID, alpha=0.7, zorder=0, linewidth=0)
        if label:
            # Anchored to the bottom: the top of the panel is reserved for the
            # legend, and overlapping the two is the classic collision.
            ax.annotate(
                name,
                xy=(start + (end - start) / 2, 0.02),
                xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=8, color=INK_MUTED,
                # Surface-coloured backing so the label stays readable where a
                # series happens to pass behind it.
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.85),
            )


def _color_for(name: str, position: int = 0) -> str:
    return SERIES_COLORS.get(name, FALLBACK_COLORS[position % len(FALLBACK_COLORS)])


def _direct_label(ax, x, y, text: str, color: str) -> None:
    """Colored dot carries identity; the text itself stays in ink.

    Keeping label text in ink rather than the series colour means the label is
    legible even where the hue is low-contrast -- which is exactly the case the
    palette validator flagged for the aqua slot.
    """
    ax.plot([x], [y], marker="o", markersize=5, color=color,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5, clip_on=False)
    ax.annotate(f"  {text}", xy=(x, y), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=INK, zorder=5,
                annotation_clip=False)


def _equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    curve = _equity(returns)
    return curve / curve.cummax() - 1.0


def _new_figure(nrows: int, height: float, ratios: list[float] | None = None,
                top: float = 0.90):
    """`top` reserves figure headroom for the title, subtitle and any legend
    placed above the axes -- set it low enough and nothing can collide."""
    fig, axes = plt.subplots(
        nrows, 1, figsize=(11, height), sharex=True,
        gridspec_kw={"height_ratios": ratios or [1] * nrows, "hspace": 0.18},
    )
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=top, left=0.095, right=0.855, bottom=0.08)
    return fig, (axes if nrows > 1 else [axes])


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_equity_with_drawdown(
    returns: pd.Series, name: str, output_dir: Path | str = DEFAULT_OUTPUT_DIR
) -> Path:
    """Equity curve above, underwater drawdown plot below, for one strategy."""
    color = _color_for(name)
    equity, drawdown = _equity(returns), _drawdown(returns)

    fig, (ax_eq, ax_dd) = _new_figure(2, 7.0, ratios=[2.2, 1.0])

    ax_eq.plot(equity.index, equity.values, color=color, linewidth=2.0, zorder=3)
    ax_eq.axhline(1.0, color=GRID, linewidth=1.0, zorder=1)
    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    _style_axis(ax_eq, "growth of $1 (log)")
    _shade_events(ax_eq, equity.index)
    # Single series: the title names it, so no legend box (dot marks the end).
    ax_eq.plot([equity.index[-1]], [equity.iloc[-1]], marker="o", markersize=5,
               color=color, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)

    ax_dd.fill_between(drawdown.index, drawdown.values, 0.0,
                       color=DRAWDOWN_FILL, alpha=0.28, linewidth=0, zorder=2)
    ax_dd.plot(drawdown.index, drawdown.values, color=DRAWDOWN_FILL, linewidth=1.2, zorder=3)
    ax_dd.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style_axis(ax_dd, "drawdown")
    _shade_events(ax_dd, drawdown.index, label=False)

    worst = drawdown.min()
    ax_dd.annotate(f"worst {worst:.1%}", xy=(drawdown.idxmin(), worst),
                   xytext=(6, 8), textcoords="offset points",
                   fontsize=9, color=INK)

    total = equity.iloc[-1] - 1.0
    fig.suptitle(f"{name.replace('_', ' ')} — equity and drawdown",
                 x=0.02, y=0.985, ha="left", fontsize=13, color=INK, weight="medium")
    fig.text(0.02, 0.94, f"total return {total:+.1%} · worst drawdown {worst:.1%}",
             ha="left", fontsize=9.5, color=INK_MUTED)

    return _save(fig, Path(output_dir) / f"equity_{name}.png")


def plot_rolling_sharpe(
    returns_by_name: dict[str, pd.Series],
    window: int = 60,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    benchmark: pd.Series | None = None,
    benchmark_name: str = "SPY",
) -> Path:
    """Rolling Sharpe for every strategy -- does the edge persist, or drift?

    A single full-sample Sharpe hides everything about *when* a strategy
    worked. This is the chart that shows a strategy living entirely off one
    good year.
    """
    fig, (ax,) = _new_figure(1, 5.0, top=0.82)

    def rolling(r):
        mean = r.rolling(window).mean()
        std = r.rolling(window).std(ddof=1)
        return (mean / std.where(std > 0) * np.sqrt(TRADING_DAYS)).dropna()

    # A window longer than the sample leaves nothing after dropna -- which
    # happens routinely on short walk-forward folds. Skip those series rather
    # than indexing into an empty array.
    if benchmark is not None:
        rb = rolling(benchmark)
        if not rb.empty:
            ax.plot(rb.index, rb.values, color=BENCHMARK_COLOR, linewidth=1.6,
                    linestyle="--", zorder=2, label=benchmark_name)

    plotted = False
    for i, (name, returns) in enumerate(sorted(returns_by_name.items())):
        rs = rolling(returns)
        if rs.empty:
            continue
        plotted = True
        color = _color_for(name, i)
        ax.plot(rs.index, rs.values, color=color, linewidth=2.0, zorder=3, label=name)
        _direct_label(ax, rs.index[-1], rs.iloc[-1], name, color)

    if not plotted:
        ax.annotate(f"not enough history for a {window}-day window",
                    xy=(0.5, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=10, color=INK_MUTED)

    ax.axhline(0.0, color=INK_MUTED, linewidth=1.0, zorder=1)
    _style_axis(ax, f"{window}-day rolling Sharpe")
    if returns_by_name:
        _shade_events(ax, next(iter(returns_by_name.values())).index)
    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False,
                           fontsize=9, ncol=4, borderaxespad=0, handlelength=1.6)
        for text in legend.get_texts():
            text.set_color(INK)

    fig.suptitle(f"Rolling {window}-day Sharpe ratio", x=0.02, y=0.985, ha="left",
                 fontsize=13, color=INK, weight="medium")
    fig.text(0.02, 0.93, "above zero means the strategy was being paid for its risk over that window",
             ha="left", fontsize=9.5, color=INK_MUTED)

    return _save(fig, Path(output_dir) / f"rolling_sharpe_{window}d.png")


def plot_comparison_dashboard(
    returns_by_name: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
    benchmark_name: str = "SPY",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """All strategies and the benchmark on shared equity and drawdown panels."""
    fig, (ax_eq, ax_dd) = _new_figure(2, 8.0, ratios=[2.2, 1.0], top=0.87)

    series = dict(sorted(returns_by_name.items()))
    if benchmark is not None:
        eq_b, dd_b = _equity(benchmark), _drawdown(benchmark)
        ax_eq.plot(eq_b.index, eq_b.values, color=BENCHMARK_COLOR, linewidth=1.6,
                   linestyle="--", zorder=2, label=f"{benchmark_name} (buy & hold)")
        ax_dd.plot(dd_b.index, dd_b.values, color=BENCHMARK_COLOR, linewidth=1.2,
                   linestyle="--", zorder=2)
        _direct_label(ax_eq, eq_b.index[-1], eq_b.iloc[-1], benchmark_name, BENCHMARK_COLOR)

    for i, (name, returns) in enumerate(series.items()):
        color = _color_for(name, i)
        equity, drawdown = _equity(returns), _drawdown(returns)
        ax_eq.plot(equity.index, equity.values, color=color, linewidth=2.0, zorder=3, label=name)
        ax_dd.plot(drawdown.index, drawdown.values, color=color, linewidth=1.4, zorder=3)
        _direct_label(ax_eq, equity.index[-1], equity.iloc[-1], name, color)

    ax_eq.axhline(1.0, color=GRID, linewidth=1.0, zorder=1)
    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    _style_axis(ax_eq, "growth of $1 (log)")
    _shade_events(ax_eq, next(iter(series.values())).index)

    ax_dd.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style_axis(ax_dd, "drawdown")
    _shade_events(ax_dd, next(iter(series.values())).index, label=False)

    legend = ax_eq.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False,
                          fontsize=9, ncol=4, borderaxespad=0, handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.suptitle("Strategy comparison", x=0.02, y=0.985, ha="left", fontsize=13,
                 color=INK, weight="medium")
    fig.text(0.02, 0.95, "net of 5bp transaction cost and 2bp slippage, positions lagged one day",
             ha="left", fontsize=9.5, color=INK_MUTED)

    return _save(fig, Path(output_dir) / "strategy_comparison.png")


def generate_all(
    returns_by_name: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
    benchmark_name: str = "SPY",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    rolling_window: int = 60,
) -> list[Path]:
    """Render every chart the README embeds."""
    paths = [
        plot_equity_with_drawdown(returns, name, output_dir)
        for name, returns in sorted(returns_by_name.items())
    ]
    paths.append(plot_rolling_sharpe(returns_by_name, rolling_window, output_dir,
                                     benchmark, benchmark_name))
    paths.append(plot_comparison_dashboard(returns_by_name, benchmark,
                                           benchmark_name, output_dir))
    return paths
