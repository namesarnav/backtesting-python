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
//   4. wait ~5 minutes -- it prints progress and downloads one file per
//      chunk of 50 tickers, so you will get ~11 files, not one
//   5. move them into the repo:
//        mkdir -p /Volumes/Quant/backtest/data/browser
//        mv ~/Downloads/yahoo_panel_*.json /Volumes/Quant/backtest/data/browser/
//   6. python -c "from engine.data_loader import DataLoader; \
//        print(len(DataLoader().ingest_browser_panel('data/browser')))"
//
// It requests 504 tickers with a 300ms gap between them -- deliberately
// unhurried, roughly what a person clicking through the site would generate.
//
// WHY CHUNKED. At 504 tickers the full payload is >100MB, which is a bad
// single Blob and an all-or-nothing download: one failure at ticker 480 and
// you refetch everything. Each chunk downloads as soon as it completes, so a
// crash costs you one chunk. To resume, set START_CHUNK below to the first
// chunk you did not get and re-run.
//
// WHY THE PAYLOAD IS TRIMMED. Yahoo's response carries metadata, dividend and
// split event tables, and pre/post market context that _parse_chart_result
// never reads. Keeping only the fields it does read cuts the download by
// roughly half without changing a single price -- the retained shape is
// exactly what the Python parser indexes into.

const TICKERS = [
  "APP", "CHTR", "CMCSA", "DIS", "ECHO", "FOX", "FOXA", "GOOG", "GOOGL", "LYV",
  "META", "NFLX", "NWS", "NWSA", "OMC", "PSKY", "RDDT", "T", "TKO", "TMUS",
  "TTD", "TTWO", "VZ", "WBD", "ABNB", "AMZN", "APTV", "AZO", "BBY", "BKNG",
  "CCL", "CMG", "CVNA", "DASH", "DECK", "DHI", "DPZ", "DRI", "EBAY", "EXPE",
  "F", "GM", "GPC", "GRMN", "HAS", "HD", "HLT", "LEN", "LOW", "LULU",
  "LVS", "MAR", "MCD", "MGM", "NCLH", "NKE", "NVR", "ORLY", "PHM", "RCL",
  "RL", "ROST", "SBUX", "TJX", "TPR", "TSCO", "TSLA", "ULTA", "WSM", "WYNN",
  "YUM", "ADM", "BF-B", "BG", "CASY", "CHD", "CL", "CLX", "COST", "DG",
  "DLTR", "EL", "GIS", "HRL", "HSY", "KDP", "KHC", "KMB", "KO", "KR",
  "KVUE", "MDLZ", "MKC", "MNST", "MO", "PEP", "PG", "PM", "SJM", "STZ",
  "SYY", "TAP", "TGT", "TSN", "WMT", "APA", "BKR", "COP", "CVX", "DVN",
  "EOG", "EQT", "EXE", "FANG", "HAL", "KMI", "MPC", "OKE", "OXY", "PSX",
  "SLB", "TPL", "TRGP", "VLO", "WMB", "XOM", "ACGL", "AFL", "AIG", "AIZ",
  "AJG", "ALL", "AMP", "AON", "APO", "ARES", "AXP", "BAC", "BEN", "BLK",
  "BNY", "BRK-B", "BRO", "BX", "C", "CB", "CBOE", "CFG", "CINF", "CME",
  "COF", "COIN", "CPAY", "EG", "ERIE", "FDS", "FIS", "FISV", "FITB", "GL",
  "GPN", "GS", "HBAN", "HIG", "HOOD", "IBKR", "ICE", "IVZ", "JKHY", "JPM",
  "KEY", "KKR", "L", "MA", "MCO", "MET", "MRSH", "MS", "MSCI", "MTB",
  "NDAQ", "NTRS", "PFG", "PGR", "PNC", "PRU", "PYPL", "RF", "RJF", "SCHW",
  "SPGI", "STT", "SYF", "TFC", "TROW", "TRV", "USB", "V", "WFC", "WRB",
  "WTW", "XYZ", "A", "ABBV", "ABT", "ALGN", "AMGN", "BAX", "BDX", "BIIB",
  "BMY", "BSX", "CAH", "CI", "CNC", "COO", "COR", "CRL", "CVS", "DGX",
  "DHR", "DVA", "DXCM", "ELV", "EW", "GEHC", "GILD", "HCA", "HSIC", "HUM",
  "IDXX", "INCY", "IQV", "ISRG", "JNJ", "LH", "LLY", "MCK", "MDT", "MRK",
  "MRNA", "MTD", "PFE", "PODD", "REGN", "RMD", "RVTY", "SOLV", "STE", "SYK",
  "TECH", "TMO", "UHS", "UNH", "VEEV", "VRTX", "VTRS", "WAT", "WST", "ZBH",
  "ZTS", "ADP", "ALLE", "AME", "AOS", "AXON", "BA", "BLDR", "BR", "CARR",
  "CAT", "CHRW", "CMI", "CPRT", "CSX", "CTAS", "DAL", "DD", "DE", "DOV",
  "EFX", "EME", "EMR", "ETN", "EXPD", "FAST", "FDX", "FDXF", "FERG", "FIX",
  "FTV", "GD", "GE", "GEV", "GNRC", "GWW", "HII", "HON", "HONA", "HUBB",
  "HWM", "IEX", "IR", "ITW", "J", "JBHT", "JCI", "LDOS", "LHX", "LII",
  "LMT", "LUV", "MAS", "MMM", "NDSN", "NOC", "NSC", "ODFL", "OTIS", "PAYX",
  "PCAR", "PH", "PNR", "PWR", "ROK", "ROL", "RSG", "RTX", "SNA", "SWK",
  "TDG", "TT", "TXT", "UAL", "UBER", "UNP", "UPS", "URI", "VLTO", "VRSK",
  "VRT", "WAB", "WM", "XYL", "AAPL", "ACN", "ADBE", "ADI", "ADSK", "AKAM",
  "AMAT", "AMD", "ANET", "APH", "AVGO", "CDNS", "CDW", "CIEN", "COHR", "CRM",
  "CRWD", "CSCO", "CTSH", "DDOG", "DELL", "FFIV", "FICO", "FLEX", "FSLR", "FTNT",
  "GDDY", "GEN", "GLW", "HPE", "HPQ", "IBM", "INTC", "INTU", "IT", "JBL",
  "KEYS", "KLAC", "LITE", "LRCX", "MCHP", "MPWR", "MRVL", "MSFT", "MSI", "MU",
  "NOW", "NTAP", "NVDA", "NXPI", "True", "ORCL", "PANW", "PLTR", "PTC", "Q",
  "QCOM", "ROP", "SMCI", "SNDK", "SNPS", "STX", "SWKS", "TDY", "TEL", "TER",
  "TRMB", "TXN", "TYL", "VRSN", "WDAY", "WDC", "ZBRA", "ALB", "AMCR", "APD",
  "AVY", "BALL", "CF", "CRH", "CTVA", "DOW", "ECL", "FCX", "IFF", "IP",
  "LIN", "LYB", "MLM", "MOS", "NEM", "NUE", "PKG", "PPG", "SHW", "STLD",
  "SW", "VMC", "AMT", "ARE", "BXP", "CBRE", "CCI", "CPT", "CSGP", "DLR",
  "DOC", "EQIX", "ESS", "EXR", "FRT", "HST", "INVH", "IRM", "KIM", "MAA",
  "O", "PLD", "PSA", "REG", "SBAC", "SPG", "UDR", "VICI", "VMRK", "VTR",
  "WELL", "WY", "AEE", "AEP", "AES", "ATO", "AWK", "CEG", "CMS", "CNP",
  "D", "DTE", "DUK", "ED", "EIX", "ES", "ETR", "EVRG", "EXC", "FE",
  "LNT", "NEE", "NI", "NRG", "PCG", "PEG", "PNW", "PPL", "SO", "SRE",
  "VST", "WEC", "XEL", "SPY",];
const P1 = 1546300800, P2 = 1704067200;   // 2019-01-01 -> 2024-01-01
const CHUNK = 50;         // tickers per downloaded file
const START_CHUNK = 0;    // bump this to resume a partial run
const GAP_MS = 300;       // pause between requests

// Keep only what engine/data_loader.py:_parse_chart_result indexes into.
function trim(json) {
  const r = json?.chart?.result?.[0];
  if (!r) return null;
  const q = r.indicators?.quote?.[0] ?? {};
  return {
    chart: {
      result: [{
        timestamp: r.timestamp,
        indicators: {
          quote: [{
            open: q.open, high: q.high, low: q.low,
            close: q.close, volume: q.volume,
          }],
          adjclose: [{ adjclose: r.indicators?.adjclose?.[0]?.adjclose }],
        },
      }],
    },
  };
}

function save(obj, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(obj)], {type: 'application/json'}));
  a.download = name;
  a.click();
}

async function grab(t) {
  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${t}`
    + `?period1=${P1}&period2=${P2}&interval=1d`
    + `&events=div%2Csplits%2CcapitalGains&includeAdjustedClose=true`;
  // No `credentials` option: cookies were shown not to matter here, and
  // sending them would require a non-wildcard CORS origin from Yahoo.
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const trimmed = trim(await r.json());
  const n = trimmed?.chart?.result?.[0]?.timestamp?.length ?? 0;
  if (n === 0) throw new Error('empty');
  return [trimmed, n];
}

const failed = [];
const short = [];
const nChunks = Math.ceil(TICKERS.length / CHUNK);
console.log(`${TICKERS.length} tickers in ${nChunks} chunks of ${CHUNK}`);

for (let c = START_CHUNK; c < nChunks; c++) {
  const slice = TICKERS.slice(c * CHUNK, (c + 1) * CHUNK);
  const out = {};
  for (const t of slice) {
    try {
      const [json, n] = await grab(t);
      out[t] = json;
      // 2019-01-01..2024-01-01 is ~1258 US trading days. Anything much
      // shorter listed late or was renamed, and will not survive the
      // full-coverage filter on the Python side -- flag it here so the
      // shrinkage is visible now rather than as a surprise row count later.
      if (n < 1200) { short.push(`${t} (${n} bars)`); console.warn(t, n, 'bars -- short history'); }
      else console.log(`${t}  ${n} bars`);
    } catch (e) {
      // One retry with a longer pause. Over 504 requests a few transient
      // failures are close to certain, and refetching a chunk is expensive.
      await new Promise(r => setTimeout(r, 2000));
      try {
        const [json, n] = await grab(t);
        out[t] = json;
        console.log(`${t}  ${n} bars (retry)`);
      } catch (e2) {
        failed.push(`${t} (${e2.message})`);
        console.warn(t, 'FAILED', e2.message);
      }
    }
    await new Promise(r => setTimeout(r, GAP_MS));
  }

  if (c === START_CHUNK && Object.keys(out).length === 0) {
    console.error('Whole first chunk failed -- stopping rather than hammering Yahoo 450 more times.');
    console.error('If this is a CORS error, make sure you are on https://finance.yahoo.com');
    break;
  }

  const name = `yahoo_panel_${String(c).padStart(2, '0')}.json`;
  save(out, name);
  console.log(`--- chunk ${c + 1}/${nChunks}: ${Object.keys(out).length}/${slice.length} tickers -> ${name}`);
}

console.log(`\nDONE. failed: ${failed.length}, short history: ${short.length}`);
if (failed.length) console.warn('FAILED (rerun these):', failed);
if (short.length) console.warn('SHORT HISTORY (will be dropped by the coverage filter):', short);
