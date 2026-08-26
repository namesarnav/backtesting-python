"""End-to-end pipeline: data -> backtest -> metrics -> charts.

This is what `docker run backtest-engine` executes, and the whole project in
one file's worth of control flow:

    DataLoader -> price panel
        -> each strategy turns the panel into a signal panel
        -> VectorizedBacktester turns signals + prices into returns
        -> metrics scores them, walk-forward re-scores out of sample
        -> the event-driven engine re-runs each one bar by bar, as a check
        -> viz renders the charts the README embeds

No arguments, no manual steps, no network: the per-ticker price cache is
committed, so a fresh clone reproduces the same numbers and the same charts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.backtest import VectorizedBacktester  # noqa: E402
from engine.data_loader import DataLoader  # noqa: E402
from engine.event_driven import available_backends, reconcile  # noqa: E402
from metrics.performance import summarize  # noqa: E402
from metrics.validation import compare_to_benchmark, walk_forward  # noqa: E402
from strategies import available_strategies, load_strategy  # noqa: E402
from viz.plots import generate_all  # noqa: E402

RESULTS_DIR = REPO_ROOT / "notebooks" / "results"

NO_DATA_HELP = """
No cached price data found, and Yahoo blocks scripted clients (HTTP 429), so
it cannot be fetched automatically.

The committed cache should have covered this -- if you are seeing it, the
cache was cleared or the universe in configs/universe.yaml was changed.

To refetch:
  1. open https://finance.yahoo.com in a browser
  2. paste scripts/yahoo_browser_fetch.js into the JavaScript console
  3. mv ~/Downloads/yahoo_panel.json data/
  4. python -c "from engine.data_loader import DataLoader; \\
                DataLoader().ingest_browser_panel('data/yahoo_panel.json')"
"""


def _load_configs() -> tuple[dict, dict]:
    backtest = yaml.safe_load((REPO_ROOT / "configs" / "backtest.yaml").read_text())
    universe = yaml.safe_load((REPO_ROOT / "configs" / "universe.yaml").read_text())
    return backtest, universe


def _format_table(rows: list[dict], benchmark_row: dict) -> str:
    columns = [
        ("strategy", "{:<16}", "{:<16}"),
        ("ann return", "{:>11}", "{:>10.2%}"),
        ("ann vol", "{:>9}", "{:>8.2%}"),
        ("Sharpe", "{:>8}", "{:>8.2f}"),
        ("Sortino", "{:>9}", "{:>9.2f}"),
        ("max DD", "{:>9}", "{:>9.1%}"),
        ("Calmar", "{:>8}", "{:>8.2f}"),
        ("OOS Sharpe", "{:>12}", "{:>12.2f}"),
        ("beta", "{:>7}", "{:>7.2f}"),
        ("turnover", "{:>10}", "{:>10.3f}"),
    ]
    header = "".join(head.format(name) for name, head, _ in columns)
    lines = [header, "-" * len(header)]

    for row in rows + [benchmark_row]:
        if row is benchmark_row:
            lines.append("-" * len(header))
        cells = []
        for key, head, cell in columns:
            value = row.get(key)
            cells.append(head.format("-") if value is None else cell.format(value))
        lines.append("".join(cells))
    return "\n".join(lines)


def _format_reconciliation(panel, close, backtest_config: dict) -> str:
    """Run every strategy through both engines and tabulate the difference.

    This is the Phase 6 checkpoint, printed on every run rather than
    asserted once: the two engines must agree on what they held (gross
    return, to ~1e-5 with costs on) and may only disagree on what it cost.

    The turnover columns are the interesting ones. The vectorized engine
    measures turnover as the change in *target* weights; the event-driven
    engine measures the change in *shares*, which also includes pulling
    drifted positions back to target. The second number is always the
    larger, and the gap is the cost the vectorized engine cannot see.
    """
    columns = [
        ("strategy", "{:<16}", "{:<16}"),
        ("gross diff", "{:>12}", "{:>12.1e}"),
        ("vec return", "{:>12}", "{:>12.2%}"),
        ("evt return", "{:>12}", "{:>12.2%}"),
        ("vec turnover", "{:>14}", "{:>14.4f}"),
        ("evt turnover", "{:>14}", "{:>14.4f}"),
        ("cost gap", "{:>10}", "{:>10.2%}"),
        ("fills", "{:>9}", "{:>9,d}"),
    ]
    header = "".join(head.format(name) for name, head, _ in columns)
    lines = [header, "-" * len(header)]

    for name in available_strategies():
        signals = load_strategy(name).generate_signals(panel)
        report = reconcile(close, signals, backtest_config)
        row = {
            "strategy": name,
            "gross diff": report["max_gross_diff"],
            "vec return": report["vectorized_total_return"],
            "evt return": report["event_driven_total_return"],
            "vec turnover": report["vectorized_avg_turnover"],
            "evt turnover": report["event_driven_avg_turnover"],
            "cost gap": report["cost_drag_diff"],
            "fills": report["n_fills"],
        }
        lines.append("".join(cell.format(row[key]) for key, _, cell in columns))

    return "\n".join(lines)


def main() -> None:
    backtest_config, universe = _load_configs()
    walk_forward_config = backtest_config["walk_forward"]
    benchmark_ticker = universe["benchmark"]

    print("=" * 100)
    print("VECTORIZED BACKTESTING ENGINE")
    print("=" * 100)

    loader = DataLoader()
    try:
        panel = loader.load_panel()
    except Exception as exc:  # noqa: BLE001 - CLI: explain rather than traceback
        print(f"\nCould not load price data: {exc}", file=sys.stderr)
        print(NO_DATA_HELP, file=sys.stderr)
        raise SystemExit(1) from exc

    close = panel["close"]
    benchmark_returns = (
        DataLoader(tickers=[benchmark_ticker])
        .load_panel()["close"][benchmark_ticker]
        .reindex(close.index)
        .ffill()
        .pct_change()
        .fillna(0.0)
    )

    print(f"\nUniverse   : {close.shape[1]} tickers, benchmark {benchmark_ticker}")
    print(f"Period     : {close.index[0].date()} to {close.index[-1].date()} "
          f"({len(close)} trading days)")
    print(f"Costs      : {backtest_config['transaction_cost_bps']}bp transaction + "
          f"{backtest_config['slippage_bps']}bp slippage, "
          f"positions lagged {backtest_config['lag_days']} day(s)")
    print(f"Allocation : {backtest_config['allocation']}")
    backends = available_backends()
    loop = "cpp (C++ extension)" if "cpp" in backends else "python (extension not built)"
    print(f"Bar loop   : {loop}")

    rows, returns_by_name = [], {}
    for name in available_strategies():
        strategy = load_strategy(name)
        result = VectorizedBacktester(backtest_config).run(close, strategy.generate_signals(panel))
        returns_by_name[name] = result.returns

        stats = summarize(result.returns, turnover=result.turnover)
        validation = walk_forward(
            panel, name, backtest_config,
            train_window=walk_forward_config["train_window_days"],
            test_window=walk_forward_config["test_window_days"],
            step=walk_forward_config["step_days"],
        )
        comparison = compare_to_benchmark(result.returns, benchmark_returns)

        rows.append({
            "strategy": name,
            "ann return": stats["ann_return"],
            "ann vol": stats["ann_volatility"],
            "Sharpe": stats["sharpe"],
            "Sortino": stats["sortino"],
            "max DD": stats["max_drawdown"],
            "Calmar": stats["calmar"],
            "OOS Sharpe": validation["oos_summary"]["sharpe"],
            "beta": comparison["beta"],
            "turnover": stats["avg_turnover"],
            "_folds": validation["n_folds"],
        })

    benchmark_stats = summarize(benchmark_returns)
    benchmark_row = {
        "strategy": f"{benchmark_ticker} buy & hold",
        "ann return": benchmark_stats["ann_return"],
        "ann vol": benchmark_stats["ann_volatility"],
        "Sharpe": benchmark_stats["sharpe"],
        "Sortino": benchmark_stats["sortino"],
        "max DD": benchmark_stats["max_drawdown"],
        "Calmar": benchmark_stats["calmar"],
        "beta": 1.0,
    }

    table = _format_table(rows, benchmark_row)
    print(f"\nRESULTS  (walk-forward: {rows[0]['_folds']} folds, "
          f"{walk_forward_config['train_window_days']}d train / "
          f"{walk_forward_config['test_window_days']}d test)\n")
    print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results_table.txt").write_text(table + "\n")

    print("\nCHARTS")
    for path in generate_all(returns_by_name, benchmark=benchmark_returns,
                             benchmark_name=benchmark_ticker, output_dir=RESULTS_DIR):
        print(f"  {path.relative_to(REPO_ROOT)}")
    print(f"  {(RESULTS_DIR / 'results_table.txt').relative_to(REPO_ROOT)}")

    reconciliation = _format_reconciliation(panel, close, backtest_config)
    print("\nENGINE RECONCILIATION  (Phase 6: vectorized vs event-driven)\n")
    print(reconciliation)
    (RESULTS_DIR / "engine_reconciliation.txt").write_text(reconciliation + "\n")
    print(f"\n  {(RESULTS_DIR / 'engine_reconciliation.txt').relative_to(REPO_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
