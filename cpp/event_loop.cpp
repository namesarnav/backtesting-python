// Phase 7: the event-driven bar loop, in C++.
//
// This is a port of `_simulate_python` in engine/event_driven.py, not a
// redesign. It is deliberately a line-for-line translation: same variable
// names, same order of operations, same conventions. That is what makes
// `tests/test_fastloop.py` able to assert the two produce the same numbers,
// and it is what makes the benchmark honest -- it measures the language and
// the loop overhead, not two different algorithms.
//
// WHY THIS LOOP AND NOT ANOTHER
// -----------------------------
// The obvious thing to port is the vectorized engine's returns aggregation.
// That would be pointless: it is already NumPy, which is already compiled
// C with SIMD, so a C++ rewrite would measure roughly 1x and prove nothing.
//
// The event-driven loop is the opposite case. It is genuinely serial -- bar
// t+1's equity depends on bar t's fills, so it cannot be vectorized away --
// and in Python each bar pays for ~15 separate NumPy calls on 40-element
// arrays. At that size the per-call dispatch overhead (allocating a
// temporary, checking dtypes, releasing the GIL, refcounting) dwarfs the
// ~40 multiply-adds of actual arithmetic.
//
// So the speedup here is not "C++ does arithmetic faster than NumPy". It is
// "C++ does not pay an interpreter tax 15 times per bar". Small arrays in a
// tight serial loop is exactly the shape where that tax dominates, which is
// why this is the right function to port and the aggregation step is not.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;

// Must match engine.event_driven.FILL_TOLERANCE. An order counts as a real
// fill when the dollars it moves exceed this fraction of the portfolio --
// a relative test, not an absolute one on the share count. See the Python
// constant for why that distinction decides whether the two backends can
// agree at all. tests/test_fastloop.py asserts the two values are equal, so
// this cannot drift out of sync silently.
static constexpr double FILL_TOLERANCE = 1e-12;

py::dict simulate(
    py::array_t<double, py::array::c_style | py::array::forcecast> prices,
    py::array_t<double, py::array::c_style | py::array::forcecast> target_weights,
    double initial_capital,
    double cost_rate,
    int lag_days,
    bool fractional_shares,
    double rebalance_threshold)
{
    // `unchecked<2>` hands back a raw accessor with no per-element bounds
    // checking. That is the whole point of being here, but it means the
    // shape checks below are load-bearing: get them wrong and this reads
    // past the end of the buffer rather than raising IndexError.
    auto P = prices.unchecked<2>();
    auto W = target_weights.unchecked<2>();

    const py::ssize_t n_bars = P.shape(0);
    const py::ssize_t n_assets = P.shape(1);

    if (W.shape(0) != n_bars || W.shape(1) != n_assets) {
        throw std::invalid_argument("prices and target_weights must have the same shape");
    }

    // See the Python docstring for why this is lag_days - 1 and not lag_days:
    // filling at the close of bar t already exposes the position to bar t+1,
    // so the loop is one bar lagged before any offset is applied.
    const py::ssize_t execution_delay = static_cast<py::ssize_t>(lag_days) - 1;

    // --- portfolio state: the two things a vectorized engine does not have
    std::vector<double> shares(static_cast<size_t>(n_assets), 0.0);
    double cash = initial_capital;

    // --- outputs, allocated once as NumPy arrays and written in place
    py::array_t<double> equity(n_bars);
    py::array_t<double> gross_returns(n_bars);
    py::array_t<double> net_returns(n_bars);
    py::array_t<double> costs(n_bars);
    py::array_t<double> turnover(n_bars);
    py::array_t<double> cash_history(n_bars);
    py::array_t<double> holdings({n_bars, n_assets});
    py::array_t<double> weights_established({n_bars, n_assets});

    auto eq = equity.mutable_unchecked<1>();
    auto gr = gross_returns.mutable_unchecked<1>();
    auto nr = net_returns.mutable_unchecked<1>();
    auto co = costs.mutable_unchecked<1>();
    auto tu = turnover.mutable_unchecked<1>();
    auto ch = cash_history.mutable_unchecked<1>();
    auto hd = holdings.mutable_unchecked<2>();
    auto we = weights_established.mutable_unchecked<2>();

    // Fills are unbounded in number, so they grow in std::vector and get
    // copied into NumPy arrays at the end.
    std::vector<int64_t> fill_bar, fill_asset;
    std::vector<double> fill_shares, fill_price, fill_notional, fill_cost;

    // Scratch space reused across bars: allocating inside the loop would
    // reintroduce exactly the per-bar overhead we came here to remove.
    std::vector<double> target_shares(static_cast<size_t>(n_assets), 0.0);
    std::vector<double> delta(static_cast<size_t>(n_assets), 0.0);
    std::vector<double> notional(static_cast<size_t>(n_assets), 0.0);

    double previous_equity = initial_capital;

    // The GIL is not needed past this point: everything below touches only
    // C++ memory and the output buffers we own. Releasing it means a caller
    // can run other Python threads while this executes.
    {
        py::gil_scoped_release release;

        for (py::ssize_t t = 0; t < n_bars; ++t) {

            // 1. Mark to market. Yesterday's shares meet today's prices --
            //    this is where the portfolio earns (or loses) the day's move.
            double equity_pre = cash;
            for (py::ssize_t i = 0; i < n_assets; ++i) {
                equity_pre += shares[i] * P(t, i);
            }

            const bool has_signal = (t >= execution_delay);
            const py::ssize_t signal_row = t - execution_delay;

            // 2/3. Decide and size. Target weight -> target dollars -> target
            //      shares, using the equity we just marked.
            for (py::ssize_t i = 0; i < n_assets; ++i) {
                const double target_w = has_signal ? W(signal_row, i) : 0.0;
                double want = (target_w * equity_pre) / P(t, i);
                if (!fractional_shares) {
                    // Truncate toward zero, never away: rounding up would
                    // quietly lever the book past 100% gross.
                    want = std::trunc(want);
                }
                target_shares[i] = want;
                delta[i] = want - shares[i];
                notional[i] = std::fabs(delta[i]) * P(t, i);
            }

            // A no-trade band: orders too small to be worth the commission
            // are skipped, and that position is left to drift.
            if (rebalance_threshold > 0.0) {
                const double floor_notional = rebalance_threshold * equity_pre;
                for (py::ssize_t i = 0; i < n_assets; ++i) {
                    if (notional[i] < floor_notional) {
                        delta[i] = 0.0;
                        target_shares[i] = shares[i];
                        notional[i] = 0.0;
                    }
                }
            }

            // 4. Fill at today's close: move the cash, pay the commission.
            double traded = 0.0;      // total notional traded this bar
            double cash_delta = 0.0;  // signed: buys drain cash, sales add
            for (py::ssize_t i = 0; i < n_assets; ++i) {
                traded += notional[i];
                cash_delta += delta[i] * P(t, i);
            }
            const double bar_cost = traded * cost_rate;

            cash -= cash_delta + bar_cost;

            double equity_post = cash;
            for (py::ssize_t i = 0; i < n_assets; ++i) {
                shares[i] = target_shares[i];
                equity_post += shares[i] * P(t, i);
            }

            gr(t) = equity_pre / previous_equity - 1.0;
            nr(t) = equity_post / previous_equity - 1.0;
            co(t) = bar_cost / previous_equity;
            tu(t) = traded / previous_equity;
            eq(t) = equity_post;
            ch(t) = cash;

            for (py::ssize_t i = 0; i < n_assets; ++i) {
                hd(t, i) = shares[i];
                // Weights just established. Exposed to bar t+1, not bar t --
                // the Python side shifts this by one before reporting it.
                we(t, i) = shares[i] * P(t, i) / equity_post;

                if (notional[i] > FILL_TOLERANCE * equity_pre) {
                    fill_bar.push_back(static_cast<int64_t>(t));
                    fill_asset.push_back(static_cast<int64_t>(i));
                    fill_shares.push_back(delta[i]);
                    fill_price.push_back(P(t, i));
                    fill_notional.push_back(delta[i] * P(t, i));
                    fill_cost.push_back(notional[i] * cost_rate);
                }
            }

            previous_equity = equity_post;
        }
    }  // GIL reacquired here

    // Copy the fill columns out. One allocation each, after the loop, rather
    // than per fill.
    auto to_array = [](const auto &vec) {
        using T = typename std::decay_t<decltype(vec)>::value_type;
        return py::array_t<T>(static_cast<py::ssize_t>(vec.size()), vec.data());
    };

    py::dict out;
    out["equity"] = equity;
    out["gross_returns"] = gross_returns;
    out["net_returns"] = net_returns;
    out["costs"] = costs;
    out["turnover"] = turnover;
    out["cash"] = cash_history;
    out["holdings"] = holdings;
    out["weights_established"] = weights_established;
    out["fill_bar"] = to_array(fill_bar);
    out["fill_asset"] = to_array(fill_asset);
    out["fill_shares"] = to_array(fill_shares);
    out["fill_price"] = to_array(fill_price);
    out["fill_notional"] = to_array(fill_notional);
    out["fill_cost"] = to_array(fill_cost);
    return out;
}

PYBIND11_MODULE(_fastloop, m)
{
    m.doc() = "C++ port of the event-driven bar loop (see engine/event_driven.py)";

    // Exposed so the test suite can assert it matches the Python constant.
    m.attr("FILL_TOLERANCE") = FILL_TOLERANCE;

    // Keyword names match the Python function exactly, so the dispatcher in
    // engine/event_driven.py can forward **kwargs to either one unchanged.
    m.def("simulate", &simulate,
          py::arg("prices"),
          py::arg("target_weights"),
          py::kw_only(),
          py::arg("initial_capital"),
          py::arg("cost_rate"),
          py::arg("lag_days"),
          py::arg("fractional_shares"),
          py::arg("rebalance_threshold"),
          "Simulate the portfolio bar by bar. Same arguments and same return "
          "dict as engine.event_driven._simulate_python.");
}
