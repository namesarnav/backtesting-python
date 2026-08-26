"""Data layer: fetch, cache, and align OHLCV panels for the backtester.

Design choices (documented per spec):

- **Panel shape**: wide DataFrame, `DatetimeIndex` rows, MultiIndex columns
`(field, ticker)` where field is one of open/high/low/close/volume. This
  (rather than a long/MultiIndex-rows panel) is what Phase 2's vectorized
  engine wants directly: `panel["close"]` slices out a plain
  `(date x ticker)` DataFrame that lines up 1:1 with a same-shaped signal
  DataFrame, so `signals * panel["close"].pct_change()`-style vectorized ops
  need no reshaping.

- **Adjustment**: split- and dividend-adjusted close (`adjclose`) is fetched
  alongside raw OHLC; open/high/low are then scaled by the per-bar ratio
  `adjclose / close`, the same approach `yfinance`'s own `auto_adjust`
  uses internally. Volume is left unadjusted (standard convention).
- **Source**: fetched via a direct HTTP call to Yahoo Finance's public chart
  API (`query2.finance.yahoo.com/v8/finance/chart/...`) rather than through
  the `yfinance` package's high-level `Ticker`/`download` calls. During
  development, `yfinance`'s crumb-authentication endpoint
  (`/v1/test/getcrumb`) was being rate-limited (HTTP 429) independent of
  this project's request volume — a known, widely-reported issue with that
  endpoint. The chart endpoint itself, hit directly, does not require a
  crumb and was reliable. This keeps `yfinance` as documented in the spec
  ("or configurable source") while avoiding a flaky dependency; see
  `_fetch_ticker_json` for the implementation.
- **Caching**: one Parquet file per ticker under `data/cache/`, keyed by
  ticker + date range + interval. A second, combined panel-level cache
  avoids re-concatenating on every run. Parquet (not CSV) preserves dtypes
  and is much faster to read back.
- **Missing data**: within a ticker's own history, small gaps (e.g. vendor
  glitches) are forward-filled up to `ffill_limit` trading days. After that,
  the panel is trimmed to the first date on which *every* ticker has data
  (`dropna` on the intersection) — i.e. we don't try to represent
  pre-IPO/pre-listing history as NaN rows inside the panel; we simply start
  the panel later. For this project's universe (long-established large
  caps), that start date is expected to equal `start_date`. This is
  documented rather than silently forward-filled so a suspicious trim shows
  up immediately instead of being masked.

Yahoo blocks scripted HTTP clients: `query1`/`query2` return HTTP 429 to
`requests` and `curl` on every network tested (cellular and residential),
with or without session cookies or the crumb handshake, while
`finance.yahoo.com` itself returns 200. The website no longer embeds
historical prices, so scraping it is not an option either. The working
route is `scripts/yahoo_browser_fetch.js`, run once from a browser console,
then `ingest_browser_panel()` below -- which shares `_parse_chart_result`
with the direct-fetch path, so prices are adjusted identically either way.
`_fetch_missing` is kept for the day Yahoo relaxes, and because a different
source can be swapped in there alone.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

FIELDS = ("open", "high", "low", "close", "volume")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_UNIVERSE_CONFIG = REPO_ROOT / "configs" / "universe.yaml"

CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _load_universe_config(path: Path = DEFAULT_UNIVERSE_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class DataLoader:
    """Fetches, caches, and aligns a multi-ticker OHLCV panel.

    Parameters
    ----------
    tickers, start, end, interval : override the universe config. If left as
        None, values are read from `configs/universe.yaml`.
    cache_dir : where per-ticker and panel-level Parquet caches live.
    ffill_limit : max consecutive trading days to forward-fill a gap within
        a single ticker's history before giving up on it.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        interval: str | None = None,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        ffill_limit: int = 5,
    ):
        cfg = _load_universe_config()
        self.tickers = tickers or list(cfg["tickers"])
        self.start = start or cfg["data"]["start_date"]
        self.end = end or cfg["data"]["end_date"]
        self.interval = interval or cfg["data"]["interval"]
        self.ffill_limit = ffill_limit

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- caching --------------------------------------------------------

    def _ticker_cache_path(self, ticker: str) -> Path:
        key = f"{ticker}_{self.start}_{self.end}_{self.interval}"
        return self.cache_dir / f"{key}.parquet"

    def _panel_cache_path(self) -> Path:
        digest = hashlib.sha1(
            "|".join(sorted(self.tickers) + [self.start, self.end, self.interval]).encode()
        ).hexdigest()[:16]
        return self.cache_dir / f"panel_{digest}.parquet"

    def _read_ticker_cache(self, ticker: str) -> pd.DataFrame | None:
        path = self._ticker_cache_path(ticker)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def _write_ticker_cache(self, ticker: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._ticker_cache_path(ticker))

    # -- fetching ---------------------------------------------------------

    def _fetch_ticker_json(self, ticker: str, retries: int = 3) -> dict:
        period1 = int(datetime.strptime(self.start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        period2 = int(datetime.strptime(self.end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        params = {
            "period1": period1,
            "period2": period2,
            "interval": self.interval,
            "events": "div,splits,capitalGains",
            "includeAdjustedClose": "true",
        }
        url = CHART_URL.format(ticker=ticker)

        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=15)
                resp.raise_for_status()
                payload = resp.json()
                result = payload["chart"]["result"]
                if not result:
                    raise ValueError(f"empty chart result for {ticker!r}")
                return result[0]
            except Exception as exc:  # noqa: BLE001 - retry on any transient failure
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))
        raise ValueError(f"failed to fetch {ticker!r} after {retries} attempts: {last_exc}")

    def _parse_chart_result(self, ticker: str, result: dict) -> pd.DataFrame:
        """Turn one Yahoo chart API result into a clean OHLCV frame.

        Split out from fetching so that JSON captured by any route -- a direct
        HTTP fetch here, or `scripts/yahoo_browser_fetch.js` running in a
        browser -- goes through identical parsing and split/dividend
        adjustment. Two copies of this logic would be two chances to adjust
        prices differently.
        """
        quote = result["indicators"]["quote"][0]
        adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
        timestamps = result.get("timestamp")
        if not timestamps:
            raise ValueError(f"No data returned for ticker {ticker!r}")

        index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
        df = pd.DataFrame(
            {
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"],
            },
            index=index,
        )
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker!r}")

        if adjclose is not None:
            adj = pd.Series(adjclose, index=index).loc[df.index]
            ratio = (adj / df["close"]).fillna(1.0)
            for field in ("open", "high", "low", "close"):
                df[field] = df[field] * ratio

        return df[list(FIELDS)]

    def _fetch_missing(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch tickers not already cached from Yahoo's chart API."""
        fetched = {}
        for ticker in tickers:
            df = self._parse_chart_result(ticker, self._fetch_ticker_json(ticker))
            self._write_ticker_cache(ticker, df)
            fetched[ticker] = df
        return fetched

    def ingest_browser_panel(self, path: str | Path) -> list[str]:
        """Populate the Parquet cache from a browser-captured JSON dump.

        Yahoo serves its chart API to browsers but returns HTTP 429 to
        scripted clients, so `scripts/yahoo_browser_fetch.js` collects the
        same JSON from inside a browser session. This reads that file and
        writes the per-ticker cache exactly as a direct fetch would, after
        which `load_panel()` works offline and nothing downstream can tell
        the difference.
        """
        payload = json.loads(Path(path).read_text())

        written = []
        for ticker, chart in payload.items():
            results = (chart or {}).get("chart", {}).get("result") or []
            if not results:
                raise ValueError(f"no chart result for {ticker!r} in {path}")
            df = self._parse_chart_result(ticker, results[0])
            self._write_ticker_cache(ticker, df)
            written.append(ticker)
        return sorted(written)

    def _load_all_tickers(self) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}
        missing = []
        for ticker in self.tickers:
            cached = self._read_ticker_cache(ticker)
            if cached is not None:
                data[ticker] = cached
            else:
                missing.append(ticker)

        data.update(self._fetch_missing(missing))
        return data

    # -- public API ---------------------------------------------------------

    def load_panel(self, use_panel_cache: bool = True) -> pd.DataFrame:
        """Return an aligned wide panel: columns MultiIndex (field, ticker).

        Guarantees no NaNs remain in the returned panel (see module
        docstring for how gaps/misalignment are resolved). Raises if a
        ticker fails to produce any usable rows.
        """
        panel_cache = self._panel_cache_path()
        if use_panel_cache and panel_cache.exists():
            return pd.read_parquet(panel_cache)

        per_ticker = self._load_all_tickers()

        panel = pd.concat(per_ticker, axis=1)  # columns: (ticker, field)
        panel = panel.reorder_levels([1, 0], axis=1).sort_index(axis=1)
        panel.columns.names = ["field", "ticker"]
        panel = panel.sort_index()

        # Forward-fill small gaps within each ticker's own history.
        panel = panel.ffill(limit=self.ffill_limit)

        # Trim to the first date every ticker has data from, then drop any
        # residual NaN rows (e.g. a gap longer than ffill_limit).
        first_valid = panel.dropna(how="any").index.min()
        if first_valid is None:
            raise ValueError("panel has no fully-populated row after ffill")
        panel = panel.loc[first_valid:].dropna(how="any")

        if panel.isna().any().any():
            raise ValueError("NaNs remain in panel after alignment — investigate source data")

        if use_panel_cache:
            panel.to_parquet(panel_cache)

        return panel


def main() -> None:
    """Fetch and cache the configured universe: `python -m engine.data_loader`.

    Populating the cache is a one-time job, and after it succeeds the rest of
    the project runs offline -- so it gets its own entry point rather than
    being something you invoke through a test.
    """
    import sys
    import time

    loader = DataLoader()
    print(f"universe : {len(loader.tickers)} tickers, {loader.start} -> {loader.end}")
    print(f"cache    : {loader.cache_dir}")

    started = time.perf_counter()
    try:
        panel = loader.load_panel()
    except Exception as exc:  # noqa: BLE001 - this is a CLI, report and exit
        print(f"\nFAILED: {exc}", file=sys.stderr)
        if "429" in str(exc):
            print(
                "\nHTTP 429 means Yahoo is rate-limiting this network's IP address, not\n"
                "you personally. Cellular/hotspot connections share one public IP across\n"
                "many subscribers (CGNAT), so the quota is usually already spent.\n"
                "Fix: connect to Wi-Fi or wired broadband and run this again.",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc

    elapsed = time.perf_counter() - started
    close = panel["close"]
    print(f"\nOK  {panel.shape[0]} rows x {close.shape[1]} tickers in {elapsed:.1f}s")
    print(f"    dates  {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"    NaNs   {int(panel.isna().sum().sum())}")
    print(f"    files  {len(list(loader.cache_dir.glob('*.parquet')))} parquet in cache")
    print("\nCache is populated -- everything from here runs offline.")


if __name__ == "__main__":
    main()
