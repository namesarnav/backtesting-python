"""Phase 7 benchmark: the event-driven bar loop, Python vs C++.

    python scripts/benchmark_cpp.py

Does two things, in this order, because the second is meaningless without
the first:

  1. **Correctness.** Runs every strategy through both backends on the real
     price panel and reports the largest disagreement. A speedup from code
     that computes something different is not a speedup.
  2. **Wall clock.** Median of N repeats after a warm-up, reported both for
     the isolated loop and for the full `run()` call.

Both numbers are given on purpose. The loop is what was ported, so the loop
speedup is what measures the port. But a user calls `run()`, which also
builds a dozen pandas objects around the loop -- constant work that C++ does
not touch, and which caps the end-to-end gain. Quoting only the loop figure
would overstate what anyone actually experiences.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.data_loader import DataLoader  # noqa: E402
from engine.event_driven import (  # noqa: E402
    EventDrivenBacktester,
    _fastloop,
    _simulate_python,
    available_backends,
)
from strategies import available_strategies, load_strategy  # noqa: E402

RESULTS_DIR = REPO_ROOT / "notebooks" / "results"
REPEATS = 9

LOOP_KWARGS = dict(
    initial_capital=1_000_000.0,
    cost_rate=7e-4,
    lag_days=1,
    fractional_shares=True,
    rebalance_threshold=0.0,
)


def _median_ms(fn, *args, repeats: int = REPEATS, **kwargs) -> float:
    """Median wall-clock ms over `repeats`, after one untimed warm-up.

    Median rather than mean: a single scheduler hiccup skews a mean and
    there is no reason to let it. The warm-up exists so the first call's
    page faults and cold caches are not charged to the measurement.
    """
    fn(*args, **kwargs)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1000.0


def _synthetic(n_bars: int, n_assets: int, seed: int = 0):
    """A price panel and a weight panel of a given size, deterministic."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(
        1.0 + rng.normal(0.0005, 0.015, (n_bars, n_assets)), axis=0
    )
    weights = rng.normal(0.0, 1.0, (n_bars, n_assets))
    weights /= np.abs(weights).sum(axis=1, keepdims=True)
    return prices, weights


def check_correctness(close, panel, config) -> list[str]:
    """Assert the two backends agree, and report by how much."""
    lines = ["CORRECTNESS  (both backends, real panel)", ""]
    lines.append(f"{'strategy':<16}{'max |return diff|':>20}{'final equity diff':>20}{'fills':>10}")
    lines.append("-" * 66)

    worst = 0.0
    for name in available_strategies():
        signals = load_strategy(name).generate_signals(panel)
        py = EventDrivenBacktester({**config, "backend": "python"}).run(close, signals)
        cc = EventDrivenBacktester({**config, "backend": "cpp"}).run(close, signals)

        return_diff = float((py.returns - cc.returns).abs().max())
        equity_diff = float(
            abs(py.equity.iloc[-1] - cc.equity.iloc[-1]) / py.equity.iloc[-1]
        )
        assert len(py.trades) == len(cc.trades), f"{name}: fill counts differ"
        # Not bit-identical, and not expected to be: NumPy sums with pairwise
        # reduction while the C++ loop accumulates in order, so the two
        # disagree in the last bit or two and that compounds mildly through
        # the equity path. 1e-10 is many orders of magnitude below anything
        # that could move a reported metric.
        assert return_diff < 1e-10, f"{name}: backends disagree by {return_diff:.2e}"

        worst = max(worst, return_diff)
        lines.append(f"{name:<16}{return_diff:>20.2e}{equity_diff:>20.2e}{len(py.trades):>10,}")

    lines += ["", f"Worst disagreement: {worst:.2e} -- float-ordering noise, "
              "not a difference in result."]
    return lines


def benchmark_real(close, panel, config) -> list[str]:
    """Time the actual workload this repo runs."""
    lines = ["", "WALL CLOCK  (real panel: 1,258 bars x 40 tickers)", ""]
    lines.append(f"{'strategy':<16}{'python':>12}{'cpp':>10}{'speedup':>10}"
                 f"{'  (full run() end to end)':>28}")
    lines.append("-" * 76)

    for name in available_strategies():
        signals = load_strategy(name).generate_signals(panel)
        weights = (
            EventDrivenBacktester(config)._allocator._target_weights(
                signals, close.pct_change().fillna(0.0)
            ).to_numpy()
        )
        prices = close.to_numpy()

        loop_py = _median_ms(_simulate_python, prices, weights, **LOOP_KWARGS)
        loop_cc = _median_ms(_fastloop.simulate, prices, weights, **LOOP_KWARGS)

        run_py = _median_ms(
            EventDrivenBacktester({**config, "backend": "python"}).run, close, signals
        )
        run_cc = _median_ms(
            EventDrivenBacktester({**config, "backend": "cpp"}).run, close, signals
        )

        lines.append(
            f"{name:<16}{loop_py:>10.2f}ms{loop_cc:>8.2f}ms{loop_py / loop_cc:>9.1f}x"
            f"{run_py:>14.1f}ms ->{run_cc:>7.1f}ms  ({run_py / run_cc:.1f}x)"
        )
    return lines


def benchmark_scaling() -> list[str]:
    """Show *why* the speedup is what it is, by varying array width.

    Bars are held fixed and only the asset count changes, so exactly one
    thing varies. The Python loop pays a fixed per-NumPy-call overhead
    roughly 15 times per bar; that cost is the same whether the arrays hold
    5 elements or 500. So the narrower the book, the more of the runtime is
    pure interpreter tax and the bigger the win from removing it.
    """
    lines = ["", "SCALING  (1,258 bars, varying width -- synthetic)", ""]
    lines.append(f"{'assets':>8}{'python':>12}{'cpp':>10}{'speedup':>10}"
                 f"{'python us/bar':>16}{'cpp us/bar':>13}")
    lines.append("-" * 69)

    for n_assets in (5, 20, 40, 100, 500):
        prices, weights = _synthetic(1258, n_assets)
        py_ms = _median_ms(_simulate_python, prices, weights, **LOOP_KWARGS)
        cc_ms = _median_ms(_fastloop.simulate, prices, weights, **LOOP_KWARGS)
        lines.append(
            f"{n_assets:>8}{py_ms:>10.2f}ms{cc_ms:>8.2f}ms{py_ms / cc_ms:>9.1f}x"
            f"{py_ms * 1000 / 1258:>16.2f}{cc_ms * 1000 / 1258:>13.2f}"
        )

    lines.append("")
    lines.append(
        "The speedup shrinks as the book widens: with 500 tickers the arrays are\n"
        "large enough that NumPy's arithmetic dominates its own call overhead, so\n"
        "there is less interpreter tax left to remove. The gain is not 'C++ beats\n"
        "NumPy at maths' -- it is 'C++ does not pay a dispatch cost 15 times a bar'."
    )
    return lines


def main() -> None:
    if "cpp" not in available_backends():
        raise SystemExit(
            "The C++ extension is not built, so there is nothing to benchmark.\n"
            "  python setup.py build_ext --inplace"
        )

    config = yaml.safe_load((REPO_ROOT / "configs" / "backtest.yaml").read_text())
    panel = DataLoader().load_panel()
    close = panel["close"]

    report = ["=" * 76, "PHASE 7 BENCHMARK: event-driven bar loop, Python vs C++", "=" * 76, ""]
    report += check_correctness(close, panel, config)
    report += benchmark_real(close, panel, config)
    report += benchmark_scaling()

    text = "\n".join(report)
    print(text)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark_cpp.txt").write_text(text + "\n")
    print(f"\nWritten to {(RESULTS_DIR / 'benchmark_cpp.txt').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
