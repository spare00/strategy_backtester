import unittest

import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

from macd_backtest_core import NY, run_relative_momentum_policy


class RelativeMomentumPolicyTests(unittest.TestCase):
    def test_app_lists_policy_and_keeps_daily_interval(self):
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"
        app = AppTest.from_file(str(app_path)).run()
        self.assertFalse(app.exception)
        self.assertIn("Monthly ETF relative momentum", app.selectbox(key="policy").options)
        self.assertIn("ARM-H5 + SHY defensive mode", app.selectbox(key="policy").options)
        self.assertIn("Monthly top-3 ETF momentum", app.selectbox(key="policy").options)
        self.assertNotIn("ETF relative momentum rotation", app.selectbox(key="policy").options)
        self.assertNotIn("Monthly sector ensemble momentum", app.selectbox(key="policy").options)
        self.assertNotIn("Monthly sector risk-adjusted momentum", app.selectbox(key="policy").options)
        self.assertNotIn("Monthly top-3 ETF momentum (always invested)", app.selectbox(key="policy").options)
        app.selectbox(key="policy").set_value("Monthly ETF relative momentum").run()
        presets = app.selectbox(key="relative_momentum_preset").options
        self.assertIn("12-month baseline (cash if none qualify)", presets)
        self.assertIn("Always invested (12-month)", presets)
        self.assertIn("Always invested (6-month)", presets)
        self.assertIn("6-month momentum + 6-month SMA filter", presets)
        self.assertIn("Risk-adjusted α=0.5", presets)
        self.assertIn("Ensemble Balanced (25/35/40)", presets)
        self.assertEqual(len(presets), 15)
        for preset in presets:
            app.selectbox(key="relative_momentum_preset").set_value(preset).run()
            self.assertFalse(app.exception, preset)
        app.selectbox(key="relative_momentum_preset").set_value("Risk-adjusted α=0.5").run()
        self.assertIn("Risk-adjusted α=0.5", app.subheader[0].value)
        app.selectbox(key="policy").set_value("Monthly top-3 ETF momentum").run()
        self.assertEqual(app.selectbox(key="top_three_preset").options,
                         ["Qualified only (cash if none)",
                          "Always invested (ignore return and SMA filters)"])
        app.selectbox(key="policy").set_value("Monthly ETF relative momentum").run()
        self.assertEqual(app.selectbox(key="bar_interval").value, "Daily")

    def test_selects_strongest_eligible_etf_at_next_session_open(self):
        dates = pd.bdate_range("2023-01-02", "2024-03-29", tz=NY)
        growth = [100.0 if stamp < pd.Timestamp("2024-01-01", tz=NY) else 200.0 for stamp in dates]
        flat = [100.0] * len(dates)

        def frame(close):
            return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=dates)

        result, trades, _ = run_relative_momentum_policy(
            {"AAA": frame(growth), "QQQ": frame(flat)}, "QQQ", 10_000, 0, 0,
            dates[0], {"AAA": [], "QQQ": []})

        first_buy = next(trade for trade in trades if trade["Side"] == "Buy")
        self.assertEqual(first_buy["Asset"], "AAA")
        self.assertEqual(first_buy["Fill time"], pd.Timestamp("2024-02-01", tz=NY))
        self.assertEqual(result.loc[first_buy["Fill time"], "rotation_symbol"], "AAA")

    def test_arm_zero_matches_always_invested_and_hysteresis_reduces_switches(self):
        dates = pd.bdate_range("2019-01-02", "2023-12-29", tz=NY)
        t = np.arange(len(dates))

        def frame(drift, amplitude):
            close = 100 * np.cumprod(1 + drift + amplitude * np.sin(t / 18))
            return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=dates)

        frames = {"QQQ": frame(0.0002, 0.001), "AAA": frame(0.0010, 0.0004),
                  "BBB": frame(0.0009, 0.0004)}
        start = pd.Timestamp("2021-01-04", tz=NY)
        baseline = run_relative_momentum_policy(frames, "QQQ", 100_000, 0, 0, start, {},
                                                 always_invested=True)
        zero_gap = run_relative_momentum_policy(frames, "QQQ", 100_000, 0, 0, start, {},
                                                always_invested=True, hysteresis_gap_pct=0)
        pd.testing.assert_series_equal(baseline[0].rotation_symbol, zero_gap[0].rotation_symbol)
        pd.testing.assert_series_equal(baseline[0].equity, zero_gap[0].equity)
        self.assertEqual(baseline[1], zero_gap[1])

        sticky = run_relative_momentum_policy(frames, "QQQ", 100_000, 0, 0, start, {},
                                               always_invested=True, hysteresis_gap_pct=5)
        self.assertLessEqual(len(sticky[1]), len(baseline[1]))
        self.assertTrue(sticky[0].invested.iloc[:-1].all())

        eligible_only = run_relative_momentum_policy(
            frames, "QQQ", 100_000, 0, 0, start, {}, candidate_symbols=("AAA", "BBB"),
            defensive_fallback_symbol="QQQ")
        eligible_hysteresis = run_relative_momentum_policy(
            frames, "QQQ", 100_000, 0, 0, start, {}, candidate_symbols=("AAA", "BBB"),
            hysteresis_gap_pct=5, defensive_fallback_symbol="QQQ")
        self.assertLessEqual(len(eligible_hysteresis[1]), len(eligible_only[1]))
        self.assertTrue(eligible_hysteresis[0].invested.iloc[:-1].all())

    def test_arm_h5_shy_fallback_and_reentry_fill_next_session(self):
        dates = pd.bdate_range("2018-01-02", "2025-12-31", tz=NY)

        def frame(anchors):
            positions = [dates.get_indexer([pd.Timestamp(day, tz=NY)], method="nearest")[0]
                         for day, _ in anchors]
            close = np.exp(np.interp(np.arange(len(dates)), positions,
                                     np.log([value for _, value in anchors])))
            return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=dates)

        frames = {
            "AAA": frame([("2018-01-02", 50), ("2020-01-02", 100), ("2021-01-04", 150),
                           ("2022-01-03", 60), ("2023-01-03", 60), ("2024-01-02", 200),
                           ("2025-12-31", 220)]),
            "SHY": frame([("2018-01-02", 100), ("2025-12-31", 105)]),
            "QQQ": frame([("2018-01-02", 100), ("2025-12-31", 120)]),
        }
        result, trades, _ = run_relative_momentum_policy(
            frames, "QQQ", 100_000, 0, 0, pd.Timestamp("2020-01-02", tz=NY), {},
            candidate_symbols=("AAA",), hysteresis_gap_pct=5,
            defensive_fallback_symbol="SHY")

        shy_fill = pd.Timestamp("2021-03-01", tz=NY)
        reentry_fill = pd.Timestamp("2023-02-01", tz=NY)
        self.assertEqual(result.loc[shy_fill, "rotation_symbol"], "SHY")
        self.assertEqual(result.loc[reentry_fill, "rotation_symbol"], "AAA")
        self.assertTrue(result.invested.loc[shy_fill:reentry_fill].all())
        self.assertTrue(any(t["Asset"] == "SHY" and t["Side"] == "Buy" and
                            t["Fill time"] == shy_fill for t in trades))
        self.assertTrue(any(t["Asset"] == "AAA" and t["Side"] == "Buy" and
                            t["Fill time"] == reentry_fill for t in trades))

    def test_arm_2x_policy_runs_with_leveraged_etf_history(self):
        dates = pd.bdate_range("2021-01-04", "2026-10-30", tz=NY)
        frames = {}
        for rank, symbol in enumerate(("SSO", "QLD", "UWM", "DDM", "ROM", "UYG", "UCC",
                                       "UXI", "UGE", "UPW", "RXL", "DIG", "UYM", "USD", "QQQ")):
            close = 100 * (1 + (rank + 1) * 0.00001) ** np.arange(len(dates))
            frames[symbol] = pd.DataFrame({"open": close, "high": close, "low": close,
                                           "close": close}, index=dates)
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"
        with patch("macd_backtest_core.fetch_ibkr_daily_bars_many", return_value=frames):
            app = AppTest.from_file(str(app_path)).run()
            self.assertIn("ARM – Hysteresis (2x ETFs)", app.selectbox(key="policy").options)
            app.selectbox(key="policy").set_value("ARM – Hysteresis (2x ETFs)").run()
            app.date_input(key="date_start").set_value(pd.Timestamp("2024-01-02").date())
            app.date_input(key="date_end").set_value(pd.Timestamp("2026-10-30").date())
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Final strategy value" in metric.label for metric in app.metric))

    def test_arm_h5_shy_policy_loads_shy_and_runs(self):
        dates = pd.bdate_range("2018-01-02", "2025-12-31", tz=NY)
        close = 100 * (1.0003 ** np.arange(len(dates)))
        frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=dates)
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"

        def fetch(symbols, _start, _end):
            self.assertIn("SHY", symbols)
            return {symbol: frame.copy() for symbol in symbols}

        with patch("macd_backtest_core.fetch_ibkr_daily_bars_many", side_effect=fetch):
            app = AppTest.from_file(str(app_path)).run()
            app.selectbox(key="policy").set_value("ARM-H5 + SHY defensive mode")
            app.date_input(key="date_start").set_value(pd.Timestamp("2021-01-04").date())
            app.date_input(key="date_end").set_value(pd.Timestamp("2024-12-31").date())
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Final strategy value" in metric.label for metric in app.metric))

    def test_risk_adjusted_momentum_prefers_lower_volatility_sector(self):
        dates = pd.bdate_range("2019-01-02", "2023-12-29", tz=NY)
        t = np.arange(len(dates))

        def frame(drift, amplitude, frequency):
            close = 100 * np.cumprod(1 + drift + amplitude * np.sin(t * frequency))
            return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=dates)

        frames = {
            "QQQ": frame(0.0002, 0.003, 0.11),
            "AAA": frame(0.0015, 0.010, 0.16),
            "BBB": frame(0.0009, 0.0015, 0.09),
        }
        start = pd.Timestamp("2021-01-04", tz=NY)
        baseline = run_relative_momentum_policy(
            frames, "QQQ", 100_000, 0, 0, start, {}, 0, candidate_symbols=("AAA", "BBB"))[1]
        self.assertEqual(next(t["Asset"] for t in baseline if t["Side"] == "Buy"), "AAA")
        for alpha in (0.5, 1.0):
            adjusted = run_relative_momentum_policy(
                frames, "QQQ", 100_000, 0, 0, start, {}, 0,
                candidate_symbols=("AAA", "BBB"), risk_adjusted_alpha=alpha)[1]
            self.assertEqual(next(t["Asset"] for t in adjusted if t["Side"] == "Buy"), "BBB")

    def test_ensemble_momentum_responds_to_recent_strength(self):
        dates = pd.bdate_range("2019-01-02", "2022-03-31", tz=NY)
        month_ends = dates.to_series().groupby(dates.to_period("M")).last().tolist()

        def frame(anchors):
            positions = [dates.get_indexer([pd.Timestamp(day, tz=NY)], method="nearest")[0]
                         for day, _ in anchors]
            values = np.exp(np.interp(np.arange(len(dates)), positions,
                                      np.log([value for _, value in anchors])))
            return pd.DataFrame({"open": values, "high": values, "low": values, "close": values},
                                index=dates)

        anchors = {
            "AAA": [(month_ends[0].date(), 70), (month_ends[24].date(), 100),
                    (month_ends[30].date(), 136.36), (month_ends[33].date(), 147.06),
                    (month_ends[36].date(), 150), (month_ends[-1].date(), 153)],
            "BBB": [(month_ends[0].date(), 70), (month_ends[24].date(), 100),
                    (month_ends[30].date(), 103.85), (month_ends[33].date(), 108),
                    (month_ends[36].date(), 135), (month_ends[-1].date(), 138)],
            "QQQ": [(month_ends[0].date(), 70), (month_ends[-1].date(), 112)],
        }
        frames = {symbol: frame(points) for symbol, points in anchors.items()}
        start = pd.Timestamp("2022-01-03", tz=NY)
        baseline = run_relative_momentum_policy(
            frames, "QQQ", 100_000, 0, 0, start, {}, candidate_symbols=("AAA", "BBB"),
            ensemble_weights=(0.0, 0.0, 1.0))[0]
        ensemble = run_relative_momentum_policy(
            frames, "QQQ", 100_000, 0, 0, start, {}, candidate_symbols=("AAA", "BBB"),
            ensemble_weights=(0.25, 0.35, 0.40))[0]
        self.assertEqual(baseline.loc[pd.Timestamp("2022-02-01", tz=NY), "rotation_symbol"], "AAA")
        self.assertEqual(ensemble.loc[pd.Timestamp("2022-02-01", tz=NY), "rotation_symbol"], "BBB")


if __name__ == "__main__":
    unittest.main()
