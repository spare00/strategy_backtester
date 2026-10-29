import math
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from macd_backtest_core import NY, run_daily_policy


POLICY = "12-month absolute momentum"


def bars(index, close):
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=index)


class MonthlyMomentumTests(unittest.TestCase):
    def test_month_end_signal_fills_next_session_and_holds_cash_after_exit(self):
        index = pd.bdate_range("2020-01-01", "2021-06-30", tz=NY)
        close = [100 if stamp < pd.Timestamp("2021-01-01", tz=NY) else
                 110 if stamp <= pd.Timestamp("2021-03-31", tz=NY) else 80 for stamp in index]
        frame = bars(index, close)
        result, trades, _ = run_daily_policy(frame, 10_000, POLICY, 0, 0,
                                              pd.Timestamp("2021-01-29", tz=NY))
        self.assertEqual([trade["Side"] for trade in trades], ["Buy", "Sell"])
        self.assertEqual(trades[0]["Signal time"], pd.Timestamp("2021-01-29", tz=NY))
        self.assertEqual(trades[0]["Fill time"], pd.Timestamp("2021-02-01", tz=NY))
        self.assertEqual(trades[1]["Signal time"], pd.Timestamp("2021-04-30", tz=NY))
        self.assertEqual(trades[1]["Fill time"], pd.Timestamp("2021-05-03", tz=NY))
        self.assertFalse(result.invested.loc[pd.Timestamp("2021-05-03", tz=NY)])

    def test_cash_yield_compounds_by_trading_session(self):
        index = pd.bdate_range("2020-01-01", "2022-01-03", tz=NY)
        frame = bars(index, [100.0] * len(index))
        start = pd.Timestamp("2021-01-04", tz=NY)
        result, trades, _ = run_daily_policy(frame, 10_000, POLICY, 0, 0, start,
                                              cash_yield_pct=5.0)
        expected = 10_000 * 1.05 ** ((len(result) - 1) / 252)
        self.assertEqual(trades, [])
        self.assertAlmostEqual(result.equity.iloc[-1], expected, places=8)

    def test_total_return_signal_includes_reinvested_distributions(self):
        index = pd.bdate_range("2020-01-01", "2021-02-10", tz=NY)
        close = [100.0 if stamp < pd.Timestamp("2021-01-01", tz=NY) else 99.8 for stamp in index]
        distribution = {"ex_date": pd.Timestamp("2020-06-01").date(), "rate": 0.5}
        frame = bars(index, close)
        result, trades, _ = run_daily_policy(
            frame, 10_000, POLICY, 0, 0, pd.Timestamp("2021-01-29", tz=NY),
            momentum_dividends=[distribution])
        self.assertEqual(trades[0]["Side"], "Buy")
        self.assertEqual(trades[0]["Fill time"], pd.Timestamp("2021-02-01", tz=NY))
        self.assertTrue(result.invested.loc[pd.Timestamp("2021-02-01", tz=NY)])

    def test_future_bars_do_not_change_past_monthly_targets(self):
        index = pd.bdate_range("2020-01-01", "2021-06-30", tz=NY)
        frame = bars(index, [100 + i * 0.05 for i in range(len(index))])
        prefix_end = pd.Timestamp("2021-03-15", tz=NY)
        start = pd.Timestamp("2020-01-01", tz=NY)
        before, _, _ = run_daily_policy(frame[frame.index <= prefix_end], 10_000, POLICY, 0, 0, start)
        changed_future = frame.copy()
        changed_future.loc[changed_future.index > prefix_end, "close"] *= 4
        changed, _, _ = run_daily_policy(changed_future, 10_000, POLICY, 0, 0, start)
        pd.testing.assert_series_equal(before.target, changed.loc[before.index].target)

    def test_app_exposes_and_runs_policy(self):
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"
        index = pd.bdate_range("2024-01-01", "2026-10-30", tz=NY)
        close = [100 + 20 * math.sin(i / 15) for i in range(len(index))]
        frame = bars(index, close)
        with patch("macd_backtest_core.fetch_daily_bars", return_value=frame), \
             patch("macd_backtest_core.fetch_corporate_actions", return_value={}):
            app = AppTest.from_file(str(app_path)).run()
            self.assertIn("Monthly 12-month absolute momentum (cash when return ≤ 0)",
                          app.selectbox(key="policy").options)
            app.selectbox(key="policy").set_value(POLICY).run()
            self.assertEqual(app.selectbox(key="bar_interval").value, "Daily")
            app.date_input(key="date_start").set_value(pd.Timestamp("2025-10-27").date())
            app.date_input(key="date_end").set_value(pd.Timestamp("2026-10-26").date())
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Final strategy value" in metric.label for metric in app.metric))


if __name__ == "__main__":
    unittest.main()
