// Fetch the whole backtest universe from Yahoo, from inside your browser.
//
// Why this exists: Yahoo serves its chart API to browsers but returns HTTP 429
// to scripted HTTP clients, so engine/data_loader.py cannot fetch directly.
// Your browser can, so we let it do the fetching and hand the result to Python.
//
// HOW TO RUN
//   1. open https://finance.yahoo.com  (must be this origin, for CORS)
//   2. View > Developer > JavaScript Console   (or Cmd+Option+J)
//   3. paste this entire file, press Enter
//   4. wait ~30s -- it prints progress, then downloads yahoo_panel.json
//   5. move that file into the repo:
//        mv ~/Downloads/yahoo_panel.json /Volumes/Quant/backtest/data/
//
// It requests 41 tickers with a 300ms gap between them -- deliberately
// unhurried, roughly what a person clicking through the site would generate.

const TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "CRM", "ADBE", "ORCL", "JPM", "BAC", "GS", "MS", "AXP", "BLK", "HON", "CAT", "UPS", "BA", "GE", "MMM", "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "XOM", "CVX", "COP", "SLB", "AMZN", "HD", "MCD", "NKE", "PG", "KO", "PEP", "WMT", "DIS", "COST", "SPY"];
const P1 = 1546300800, P2 = 1704067200;   // 2019-01-01 -> 2024-01-01

const out = {}, failed = [];
let checked = false;
for (const t of TICKERS) {
  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${t}`
    + `?period1=${P1}&period2=${P2}&interval=1d`
    + `&events=div%2Csplits%2CcapitalGains&includeAdjustedClose=true`;
  try {
    // No `credentials` option: cookies were shown not to matter here, and
    // sending them would require a non-wildcard CORS origin from Yahoo.
    const r = await fetch(url);
    if (!r.ok) { failed.push(`${t} (HTTP ${r.status})`); console.warn(t, r.status); }
    else {
      const j = await r.json();
      const n = j?.chart?.result?.[0]?.timestamp?.length ?? 0;
      if (n === 0) { failed.push(`${t} (empty)`); console.warn(t, 'empty'); }
      else { out[t] = j; console.log(`${t}  ${n} bars`); }
    }
  } catch (e) { failed.push(`${t} (${e.name})`); console.warn(t, e); }
  if (!checked) {
    checked = true;
    if (!out[t]) {
      console.error('First ticker failed -- stopping rather than hammering Yahoo 40 more times.');
      console.error('If this is a CORS error, make sure you are on https://finance.yahoo.com');
      break;
    }
  }
  await new Promise(r => setTimeout(r, 300));
}

console.log(`\nDONE: ${Object.keys(out).length}/${TICKERS.length} tickers`);
if (failed.length) console.warn('FAILED:', failed);

const a = document.createElement('a');
a.href = URL.createObjectURL(new Blob([JSON.stringify(out)], {type: 'application/json'}));
a.download = 'yahoo_panel.json';
a.click();
console.log('Saved yahoo_panel.json to your Downloads folder.');
