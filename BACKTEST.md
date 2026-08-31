# How I built a backtesting engine

I want to walk through this the way I'd explain it to someone sitting next to
me — in the order I actually built it, including the parts where I was wrong
for a while.

The short version of what it is: you give it five years of daily stock
prices and a trading rule, and it tells you what would have happened if
you'd followed that rule. There are two separate engines that compute the
same thing in completely different ways, three strategies, and a C++
extension. The whole thing runs with `docker run` and no network.

The short version of what I found: none of the three strategies beat just
buying SPY, once you measure them properly. I'll get to why that's the good
outcome.

## The thing I was worried about from the start

The arithmetic of a backtest is genuinely trivial. Multiply what you held by
what it returned, add it up across your stocks, compound it. Four lines of
pandas and you have an equity curve.

That's the trap. The code looks finished long before it's correct, and
here's the part that bothered me: every way it can be wrong makes the
results look *better*. If you accidentally let the strategy see tomorrow's
price, you don't get an exception — you get a beautiful upward curve. If you
pick which stocks to trade by looking at the whole five years first, same
thing. If you forget to charge trading fees, same thing. There's no error
message for any of it. The program runs happily and hands you a number that
is fiction.

So I decided upfront that the project wasn't really "build a backtester." It
was "build a backtester I can't fool myself with." Every one of those four
failure modes gets a specific mechanism to prevent it, and every mechanism
gets a test that screams if someone removes it. That framing drove basically
every decision after it.

I also drew a hard line around scope. Daily bars, US stocks, a few dozen
names, flat percentage trading costs. No intraday data, no order book, no
options, no live trading, and no machine learning. Every one of those is a
whole project, and adding a weak version of one would have made everything
else less trustworthy rather than more.

## Getting the data, and the week Yahoo told me no

I started with the boring part, which turned out not to be boring.

The first real decision was the *shape* of the data, and I want to dwell on
it because it quietly determines everything downstream. Stock price data can
be laid out a few ways. You can have one row per (date, stock) — the "long"
format, which is what a database would hand you. You can have a dictionary
mapping each ticker to its own little table. Or you can have one big wide
table with dates going down and stocks going across.

I went with the wide table, and the reason is that it makes the central
calculation vanish. If my prices are a wide table of dates × stocks, and I
build my trading signals as *exactly the same shape*, then "what I held
times what it returned" is one multiplication across the entire five-year
history at once. No looping over dates. No looping over stocks. No
regrouping.

The dictionary-of-tables version would have forced a Python loop over
tickers into every single calculation, which is precisely the thing the
engine exists to avoid. The long format would have needed a group-by on
every operation. Both work; both are slower and read worse. The cost of the
wide-table choice is that everything downstream has to conform to it — if a
strategy wants to hand back a different shape, the strategy changes, not the
engine.

Then there's a finance detail that will silently wreck you if you skip it.
Raw closing prices are useless. If a stock does a 2-for-1 split, the price
halves overnight, and if you naively compute the return you'll record a 50%
loss that never happened. Same with dividends. So the loader pulls the
split- and dividend-adjusted close, and scales the other price fields by the
same ratio so they stay consistent with each other.

Caching I did in two layers, which sounds like over-engineering until you
think about why. One Parquet file per ticker, plus one combined file for the
assembled panel. Two layers because they solve different annoyances: the
per-ticker files mean that adding one stock to my universe re-downloads one
stock rather than all forty-one, and the combined file means that just
re-running the thing doesn't re-do the assembly work. Parquet rather than
CSV because it remembers what a date is instead of making me re-parse
strings every time.

And then the actual downloading fell apart.

The plan was to hit Yahoo Finance's public price API with `requests`. First
the `yfinance` library's normal calls started failing — its authentication
endpoint was handing back HTTP 429 "too many requests" regardless of how
little I was asking for, which turns out to be a known problem with that
specific endpoint. Fine, I thought, I'll skip the library and call the price
endpoint directly, since that one doesn't need the auth step.

That worked. Then it stopped working. 429 again, on every network I tried —
my home wifi, my phone's hotspot, a different connection entirely. With
cookies, without cookies, with the auth handshake, without it. Meanwhile I
could open finance.yahoo.com in Chrome and see the exact data rendering
perfectly.

That's when I understood the situation: Yahoo wasn't rate-limiting *me*.
It was declining to serve scripts at all, while happily serving browsers.
And the modern site doesn't embed the price history in the page HTML
anymore, so scraping the page wasn't a fallback either.

The fix is unglamorous and completely reliable. I wrote a small JavaScript
file that you paste into the browser's own console on Yahoo's site. Run
there, the requests are indistinguishable from the page's own requests,
because they *are* the page's own requests. It dumps a JSON file, and the
loader has a function that reads it and pushes it through the same parsing
and adjustment code as the direct download path — so however the prices
arrived, they get treated identically.

Then I committed the cached data to the repo. About 2.8MB of Parquet: 40
stocks plus SPY, January 2019 through December 2023, 1,258 trading days,
zero missing values.

I want to flag that as a decision rather than a workaround, because I think
it's the right call independent of the blocking. A project that claims to be
reproducible cannot depend on somebody else's API still existing. With the
data committed, someone cloning this in two years gets identical numbers on
a machine with no internet. The download path is still in there for the day
Yahoo relaxes, and it's the one place a different data vendor would need to
be plugged in.

The real lesson was that the data layer is where the unglamorous time goes,
and that "it worked on my machine last Tuesday" is not a data source.

## The engine itself

The core is about 290 lines, and honestly most of that is comments.

It takes prices and signals, and it does this in this order: check the two
line up, turn the raw signals into actual portfolio weights, **lag them by a
day**, measure how much trading that implies, charge fees on that trading,
multiply positions by returns and sum across stocks, compound.

A few of those steps deserve explanation.

Checking the inputs line up sounds like defensive boilerplate. It isn't.
pandas will cheerfully take two tables whose dates don't quite match and
align them for you, filling the gaps with nothing. You won't get an error —
you'll get a perfectly plausible equity curve computed from a portfolio that
was accidentally sitting in cash a third of the time. So the engine refuses
to run rather than guess. A loud failure beats a quiet wrong answer, and
that's a principle I applied everywhere.

The signals a strategy produces aren't final weights, they're *desires*. A
strategy says "+1, I want to be long this" or "−1, short this" or "0,
nothing." The engine takes those and normalizes them into weights that add
up to a fully-invested portfolio. There's a subtlety in how you normalize: I
divide by the sum of the *absolute* values, not the plain sum. In a
portfolio with equal longs and shorts, the plain sum is zero, and dividing
by zero produces infinities. Using absolute values means a market-neutral
book comes out at 100% invested and 0% net exposure — which is exactly what
"market neutral" is supposed to mean.

Fees get charged on turnover — how much the portfolio changed today, summed
across positions, times 7 basis points (5 for commission, 2 for slippage).
Every bit of turnover is something a real broker really bills you for. One
of my three strategies lives or dies entirely on this line, which I'll get
to.

And then the actual portfolio calculation is one expression:

```python
gross_returns = (positions * asset_returns).sum(axis=1)
```

That's the payoff for the data-shape decision. Whole history, both
dimensions, one line.

There are zero `for` loops in that entire file. Here's a problem with that,
though: if someone later added a slow Python loop, the numbers would be
*identical*. Nothing would break. No test would notice. So I wrote one that
parses the file itself and fails if a loop shows up anywhere in it.
"Vectorized" is a claim about the code, so I made the test look at the code.

## The one line the whole thing rests on

This is the part I'd want to be asked about.

A signal computed from today's closing price cannot be traded today. The
closing price doesn't exist until the market has closed. By the time you
know it, the opportunity to act on it at that price is gone. So the engine
holds *yesterday's* target:

```python
return weights.shift(self.lag_days).fillna(0.0)
```

One line. And it's easy to underrate it until you see the size of the
effect, so I built a tiny test dataset specifically to show it. Two stocks,
three days. On day 1, stock A jumps 10%. On day 2, stock B jumps 10%. The
strategy is a caricature: it buys whatever just moved.

Run that without the lag and it returns **+20.77%**. Run it with the lag and
it returns **−0.07%**. Same data, same rule, one `shift()`.

That's what look-ahead bias buys you. A losing strategy becomes a 21% winner
and nothing crashes, nothing warns you, and the equity curve is smooth and
lovely. Those two exact numbers are asserted in the test suite, so if
anybody ever deletes that line the failure is unmissable. And the engine
flatly refuses to run with a lag of zero — it's not offered as an option.

There are two sneakier versions of the same bug that I had to handle
separately.

The first: one of the position-sizing modes weights stocks by their recent
volatility, so that a calm stock and a wild one contribute similar risk. But
volatility is computed from prices — so if you lag the *signal* but compute
the sizing from today's data, you've leaked information right back in
through the side door. The sizing is therefore folded into the weights
*before* the lag, so a single `shift()` covers both.

The second is worse, and it took me a while to fully appreciate it. It shows
up in the pairs strategy, so let me explain it there.

## Three strategies

All three implement the same single method — hand it prices, it hands back
desired positions — and the engine genuinely doesn't know which one it's
running. Adding a fourth is one new file and one config entry.

**Momentum** is the oldest documented pattern in stocks: things that have
been going up tend to keep going up, at least for a few months. Every day I
measure each stock's return over the last 126 trading days (about six
months), rank all forty against each other *on that day*, and buy the top
10%.

The word "against each other" is doing a lot of work. The comparison is
stock-versus-stock on the same day, never stock-versus-its-own-past. That
means in a market where everything dropped 30%, this still holds the names
that dropped least — it's a bet on relative strength, which is a very
different and much more hedgeable claim than "this will go up." There's also
a bug I nearly wrote here: ranking across the wrong axis. Rank across the
row and you're comparing stocks to each other today. Rank down the column
and you're comparing today's momentum to last year's momentum, which is a
completely different strategy that happens to run without complaint.

**Mean reversion** is the opposite bet on a much shorter horizon: a stock
that's shot away from its own recent average tends to snap back. Both
effects are real, they just operate on different timescales, which is why
you can run them side by side.

The measure is how many standard deviations the price sits from its 20-day
average. If a stock normally trades around $100 and typically wiggles $2,
then $104 is two standard deviations rich and $97 is one and a half cheap.
Dividing by the wiggle size is what makes it comparable across stocks — a $4
move in a sleepy utility and a $4 move in a volatile tech name are not the
same event, but "two sigma" means the same thing in both.

Worth noting the sign, because getting it backwards is easy and produces a
strategy that still runs: a *high* score means the stock is expensive, which
means **short** it. The tests pin the direction down for exactly that
reason.

One design detail I'm happy with: the threshold to open a position and the
threshold to close it are different — open at 1.0 sigma, close only once it
comes back inside 0.25. With a single threshold, a stock hovering right at
the line would flip in and out of the portfolio every other day, and since
the engine charges fees on every change, that flickering is expensive. The
gap between the two numbers is a deliberate brake.

**Pairs trading** is the interesting one. The idea is that some stocks are
economically tied together — two banks, two refiners — so while each
wanders unpredictably on its own, the *gap* between them stays fairly
stable. When the gap stretches unusually wide, you bet on it closing: buy
the one that lagged, short the one that ran. If it works, it works
regardless of what the overall market does, which is the appeal.

The statistical machinery for "these two wander together" is called
cointegration, and there's a standard test for it that regresses one stock
on the other and asks whether the leftover residual behaves in a
mean-reverting way. That residual is what you actually trade. The regression
also gives you the hedge ratio — how many shares of one offset a share of
the other, so the market exposure cancels.

With forty stocks there are 780 possible pairs, and running the full
statistical test on all of them is slow, so I pre-screen by correlation
(cheap) and only run the real test on the ten most promising.

And here's the sneaky look-ahead problem I promised.

The obvious way to build this is: test all 780 pairs over the whole five
years, find the most cointegrated one, then backtest that pair over those
same five years. It produces a gorgeous chart. It is also completely
worthless, because you *chose the pair using knowledge of how it turned
out*. You could not have known in January 2019 which pair was going to stay
glued together through 2023.

What makes this nastier than a missing `shift()` is that every single day's
arithmetic is still correct. Nothing in the daily calculation looks wrong,
because nothing in the daily calculation *is* wrong. The leak is entirely in
the choice of what to trade. It's look-ahead bias wearing a lab coat.

The fix is a formation window: the first 252 days are used only to pick the
pair and fit the hedge ratio, and no trading happens during them. Everything
after is honest with respect to that choice. I'll show in a minute what that
was worth.

## Making the scoring honest

Once you have returns, you need to turn them into numbers you can compare,
and there are a couple of conventions here that are easy to get subtly
wrong.

Annualizing uses 252 days, not 365, because your data only has entries for
days the market was open. Sharpe and Sortino get computed arithmetically —
average daily return over daily standard deviation, scaled up — which is
what everyone else does, while the headline return figure is computed
geometrically, because compounding is what actually happened to the money.
Those two are different numbers, and quietly mixing them gives you a Sharpe
that disagrees with the rest of the world's.

There's a small thing I'm glad I handled: infinite Sharpe ratios. You get
one by dividing by a standard deviation of zero, which happens whenever a
strategy simply didn't trade over the window you're measuring — and that
happens constantly once you start slicing history into small pieces. Every
ratio in my metrics returns "undefined" rather than infinity in that case.
Infinity would quietly top a leaderboard and poison any average it landed
in; undefined is the truthful answer.

But the metric that actually matters is the one that comes from
walk-forward validation, and this is the piece that changed my conclusions.

Here's the problem it solves. A single backtest over five years tells you
how a strategy did on *one* sample of history — and every choice I made
while building it was made by someone who already knew what happened in that
history. That's me. I picked a six-month momentum window and a 20-day
reversion window and a 1.0-sigma threshold, and even if I never explicitly
tuned them, I chose them with a general sense of what has worked in markets
I've read about. The backtest can't see that bias, but it's there.

Walk-forward attacks it by repeatedly splitting time. Train on two years,
then test on the following six months — data the choices never saw. Slide
forward, repeat. Five folds. Then stitch all the test windows together and
that's your out-of-sample track record, which is a far better guess at real
performance than any full-sample number.

The gap between the in-sample and out-of-sample numbers is the real output.
A strategy that scores 2.5 in-sample and 0.1 out-of-sample isn't a good
strategy that got unlucky. It's a curve fit.

Two implementation details do the real work. Each fold hands the strategy
*only* the slice of prices from that fold, never the full history. That
matters for two reasons: momentum needs 126 days of warm-up before it
produces anything, so it needs to see the training window to be ready when
the test window starts — and the pairs strategy picks its pair from the
first chunk of whatever you hand it, so slicing per fold means the pair gets
selected inside the training period, out of sample with respect to the test.
Hand it the whole panel and the pair is frozen at 2019 forever, and you've
leaked.

## What came out

Forty liquid US stocks across six sectors, 2019 through 2023, after fees,
positions lagged a day.

| strategy | ann return | ann vol | Sharpe | max DD | **out-of-sample Sharpe** | beta |
|---|---:|---:|---:|---:|---:|---:|
| mean reversion | −13.33% | 18.05% | −0.70 | −51.2% | **−0.89** | 0.39 |
| momentum | 19.40% | 27.45% | 0.78 | −32.3% | **0.90** | 0.88 |
| pairs | 2.80% | 5.82% | 0.50 | −11.3% | **−0.25** | −0.02 |
| *SPY buy & hold* | *15.60%* | *20.99%* | *0.80* | *−33.7%* | — | *1.00* |

Momentum is the one that fools people, and it nearly fooled me. It made
19.4% a year against SPY's 15.6%. That looks like a win, and if I'd stopped
at that column I'd have written it up as one. But look one column over: it
did that with 27.5% volatility against SPY's 21.0%. Per unit of risk taken,
it's *worse* than the index. And its beta of 0.88 says most of that return
is just market exposure, which anyone can buy for free by holding SPY. The
honest description is "a leveraged index fund with extra trading costs."

I think this is the single most common way a student backtest lies to its
author. The return number is bigger, so it reads as skill. It isn't skill,
it's leverage plus fees.

Pairs is the cautionary tale, and it's the one that justifies all the
walk-forward machinery. Over the full sample it scores 0.50 — a modest,
believable success. Out of sample it scores **−0.25**. That entire gap is
the value of having picked the pair after seeing the whole period.
Re-select the pair on each training window, trade the next six months blind,
and the edge evaporates. Any pairs backtest that doesn't do this is
reporting fiction, and mine would have been too.

Mean reversion isn't wrong so much as it's eaten alive. It trades 33,751
times, roughly 3.9% of capital a year in fees before it's been right or
wrong about anything. Turn the cost model off and it looks respectable.
That's the whole argument for having a cost model.

And pairs is the only genuinely market-neutral one — beta of −0.02, worst
drawdown of 11% against everyone else's 32% to 51%. It delivers exactly the
risk profile it advertises. It just doesn't make money.

So: three strategies, three completely different reasons for not working.
One is beta in disguise, one is selection bias, one is transaction costs. I'd
rather report that than a Sharpe of 2.5 I couldn't defend, and I want to be
clear that the 2.5 was genuinely available — delete the lag, pick the pair
on the full sample, turn off fees. Three edits, all one line each.

For the charts I made a couple of deliberate choices worth mentioning.
Equity curves are on a log scale, because on a linear axis the later years
look more dramatic purely because the numbers are bigger, and I want equal
percentage moves to look equal. Each strategy's chart shows its drawdown
underneath the equity line, because the curve alone hides what it felt like
to hold — and duration of a drawdown, not just depth, is what determines
whether a real person would have stuck with it. And there's a rolling
60-day Sharpe chart that I mostly find humbling: every strategy swings
between roughly −6 and +7 depending on which two months you look at. Short
windows are extremely noisy, and "it's been working lately" means almost
nothing.

## Then I built the whole engine again

At this point the thing worked. I built a second, completely different
engine anyway, and this is the part I'd most want to talk about.

The first engine thinks in weights and computes the entire history in a
handful of whole-table operations. The second one *simulates*: one day at a
time, holding an actual cash balance and an actual share count for each
stock, placing orders that fill at the closing price. Bar arrives, state
updates, orders go out.

The point wasn't redundancy. It was to find out what the fast one was
quietly assuming, because a thing that's fast and wrong is worse than a
thing that's slow.

The design decision that makes the comparison mean anything: the
event-driven engine *calls the vectorized engine's weight calculation*. They
literally cannot disagree about what to hold. So every difference between
their outputs is attributable to *how* it gets held, rather than being two
different portfolios doing two different things — which would tell me
nothing.

With fees switched off, the two produce identical returns to about fifteen
decimal places over 1,258 days. That's the assertion I care about, because
gross return has no modelling choices in it. If it diverged, one of them
would have a bug in the lag or the weighting.

Getting there meant fixing an off-by-one that I nearly shipped.

The obvious thing to write in a simulation loop is "today, go get the target
from `lag_days` ago." That's wrong, and it took me a bit to see why. If you
fill your orders at *today's close*, then the position you just established
is exposed to *tomorrow's* move. The simulation is already lagged by a day,
structurally, just from how it works. So a configured lag of 1 has to map to
*zero* extra delay in the loop.

Write the obvious version instead and you double-lag the entire portfolio,
forever. Nothing crashes. The equity curve looks fine. The two engines just
quietly disagree by one day for all eternity. I caught it by reasoning
through the convention on the vectorized side before writing any tests, and
there's now a test that pins the two engines' positions together so it can
never drift back.

And here's what the comparison actually exposed, which I did not predict.

The event-driven engine reports *more* trading than the vectorized one, for
all three strategies, and correspondingly lower returns. The biggest reason
is something I'd call a blind spot rather than a bug.

The vectorized engine measures trading as the change in *target weights*. So
if your strategy says "hold 5% of AAPL" every day forever, it measures zero
trading after the initial purchase. And that's just not true. Holding a
constant *weight* is not holding a constant *position* — prices move
overnight, your 5% drifts to 5.3%, and pulling it back to 5% is a real trade
that a real broker really charges you for. The simulation tracks shares, so
it sees those trades. The vectorized version structurally cannot.

The place this bites hardest is pairs, where measured trading jumps 46%.
Which makes sense once you see it: pairs holds a nearly static two-stock
position, so almost *all* of its real trading is drift correction — exactly
the category the fast engine is blind to. The strategy that looked cheapest
to trade is the one whose costs were most understated. That's the finding,
and I want to be honest that it fell out of the reconciliation rather than
being something I went looking for.

There are two smaller differences I chased down and decided to keep. Fees
land a day earlier in the simulation, because a trade decided at today's
close is filled at today's close, so the cash leaves today — whereas the
vectorized engine charges it against tomorrow, when the position takes
effect. That's a one-day shift, immaterial to the total. And orders get
sized on pre-commission equity, because you don't know the commission until
the order exists; that's what a live system does, and it's why the two agree
*exactly* only when fees are zero.

None of this reverses a conclusion. Momentum still loses on risk-adjusted
terms, mean reversion is still bad, pairs still fails out of sample. What
changed is that I now know the cost estimate from the fast engine is biased
optimistic, and biased *more* for low-turnover strategies than high-turnover
ones — which is the opposite of what I'd have guessed.

The broader thing I took away: real trading systems are event-driven not
because they're more accurate, but because in production your backtest and
your live trader have to be *the same code*. A live trader is inherently an
event loop. A vectorized backtest can't be run live at all — it needs the
entire future sitting in a table before it can compute anything. It's the
right tool for research throughput, and it will lie to you about execution,
quietly, in the flattering direction.

## The C++ part

I wanted a compiled component, but I didn't want a fake one, so the first
question was what to port.

The tempting target is the vectorized engine's main calculation. That would
have been a wasted weekend: it's already NumPy, which is already compiled C
with vector instructions underneath. An honest measurement would have come
back at roughly 1x, and I'd have had a chart showing my C++ tied with
somebody else's C.

The simulation loop is the opposite situation, and this is why building it
first mattered. It's irreducibly serial — tomorrow's equity depends on
today's fills, so you can't vectorize it away — and in Python each day pays
for about fifteen separate NumPy calls on 40-element arrays. At that size,
the *overhead* of each call (allocate a temporary, check the types, manage
reference counts, return an object) completely dwarfs the forty
multiply-adds of actual arithmetic. That's real, removable waste.

So the C++ file is a line-for-line translation of the Python loop. Same
variable names, same order of operations. Deliberately boring, because I
want to be measuring the language, not comparing two different algorithms.

Correctness came before any timing, since a speedup from code that computes
something else isn't a speedup. Across all three strategies on real data,
the largest disagreement in daily returns is about 8e-15, with identical
trade counts — that's floating-point ordering noise, because NumPy sums in
pairs and my loop sums in order, so they differ in the last bit.

The loop itself came out 28 to 47 times faster depending on the strategy.
But I report a second number alongside it, because the honest end-to-end
figure is 5 to 10x: a user calls the full function, which also builds a
dozen pandas objects around the loop that C++ never touches. Quoting only
the 47x would overstate what anyone actually experiences.

The result I like most is the scaling table. Holding the number of days
fixed and varying only how many stocks are in the portfolio:

```
  assets      python       cpp   speedup
       5      9.85ms    0.06ms    168.2x
      20     17.86ms    0.29ms     61.8x
      40     28.25ms    0.70ms     40.4x
     100     61.28ms    1.87ms     32.8x
     500    311.68ms   14.57ms     21.4x
```

The speedup *shrinks* as the portfolio widens, and that's the whole
explanation of the result. Python pays the same fixed per-call overhead
whether the array has 5 elements or 500 — so the narrower the portfolio, the
more of the runtime is pure interpreter tax, and the more there is to
delete. By 500 stocks the arrays are finally big enough that NumPy's actual
arithmetic dominates, and the gap closes to 21x.

So the claim isn't "C++ beats NumPy at maths." It's "C++ doesn't pay a
dispatch tax fifteen times a day." A benchmark that just said "47x faster!"
would have hidden the entire mechanism, and it's the mechanism that's
interesting.

Two things had to change to make the port work, and both turned out to be
improvements to the *Python* side.

The first was a genuine design flaw that the port surfaced. My original loop
counted an order as real if the change in share count exceeded a tiny
absolute threshold. That's not scale-free — a tiny fraction of a share of a
$1 stock and of a $1,000 stock are not the same event — and it broke.
Specifically: hold a single stock at 100% and the target share count is
*exactly* what you already own, so every day's difference is floating-point
dust sitting right on the threshold, tipping one way or the other depending
on summation order. The two engines disagreed on the trade count by one. The
diagnosis matters more than the fix: this wasn't a porting bug, it was a bug
in code I'd already written and tested, which the port dragged into the
light. Measuring the order as a *fraction of the portfolio* instead puts
that dust three orders of magnitude clear of the line.

The second was subtler. My equivalence tests failed on the cash balance
differing by 5e-10, and my first instinct was that I'd mistranslated
something. I hadn't. Cash is computed as a residual — total equity minus
everything you're holding — and in a long/short portfolio those are two
numbers around $2 million that nearly cancel to a balance of a few dollars.
So cash keeps absolute precision at the scale of its *inputs* while its
relative precision is destroyed. That's textbook catastrophic cancellation
behaving exactly as advertised. The tests now compare dollar amounts in
dollars and ratios relatively, because a single tolerance for both would
either fail on cash forever or wave through a real divergence in returns.

The loop also releases Python's global interpreter lock, so two backtests
can genuinely run on two threads. There's a test that asserts this by
*timing* it, because if someone removed the release, nothing else in the
suite would notice.

## About the tests

There are 102 of them and they run in about three seconds, but the count
isn't the point. Every one exists because some specific wrong answer would
otherwise have looked right.

Some check arithmetic against numbers I worked out by hand — Sharpe,
drawdown, and a simple buy-and-hold portfolio, all verified against manual
calculation rather than against themselves.

Some exist to catch a silent lie. The toy dataset asserts both the +20.77%
and the −0.07%. Another one rewrites the *future* of a price series and
demands that every past signal come back byte-identical, which is a direct
assault on look-ahead that no amount of reading the code can substitute for.

Some check things nothing else could see: the loop-detector that parses the
engine's own source, the GIL-release timing test. Both guard properties
where the wrong behavior produces identical output.

And one pair of assertions encodes a decision rather than a behavior: the
two engines must match *exactly* on gross returns with fees off, and are
permitted to differ slightly with fees on. That's my modelling stance
written down as an executable statement — they may disagree about cost,
never about what was held.

The rest are the degenerate cases. A strategy that never trades. A window
longer than the data. A flat-lined stock with zero volatility. A universe
where no pair cointegrates. Signals that are all zeros. Every one of those
is a place where a reasonable-looking implementation divides by zero and
poisons something downstream.

## Wrapping up

The whole thing ships as a Docker image that reproduces the table, the
engine comparison, and all the charts with no arguments and no network,
compiling the C++ during the build so the container takes the fast path. CI
lints it, builds the extension, runs the tests on three Python versions,
runs the pipeline end to end, and separately builds and runs the image.
Every number in the writeup is generated by the code that computes it, so
nothing is typed by hand.

There's a list of limitations in the README that I'd rather state than have
someone find. The big one is survivorship bias — my forty stocks are forty
companies that exist *today*, so anything that went bankrupt between 2019
and 2023 simply isn't in the sample, which flatters every result including
the benchmark. Costs are a flat percentage with no market impact. Shorts are
assumed free to borrow, which they aren't, especially for the names mean
reversion wants to short. And Sharpe uses a 0% risk-free rate over a period
when rates went from 2.4% to 5%, which makes every risk-adjusted number here
optimistic.

If I kept going, the first thing I'd change is re-selecting the pairs on a
rolling basis instead of once — that's the change most likely to actually
move a conclusion, since I don't currently know whether pairs failed because
the idea is bad or because one pair decoupled. After that, a
survivorship-free universe, and market impact in the cost model, which would
penalize exactly the strategies that currently look cheapest.

Three things ate most of the time, and all three turned out to be the actual
content rather than obstacles to it. Getting the data, because a source you
don't control can just stop working. Proving there's no look-ahead, because
you can't do that by reading code — you have to produce a number that moves
by 21% when you delete one line. And building the engine twice, because the
second implementation is the only way to find out what the first one was
assuming. The 46% understatement on pairs wasn't a hypothesis I tested. It
fell out.

The conclusion I'd defend hardest is the boring one: nothing here beats
buying the index. A much better-looking version of this project was three
one-line edits away the entire time. Most of the work was in not making
them.

---

*Code: [github.com/namesarnav/backtesting-python](https://github.com/namesarnav/backtesting-python)*
