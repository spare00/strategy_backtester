import math
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from macd_backtest_core import rsi_indicators, run_rsi_policy


def bars(close):
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close},
                        index=pd.date_range("2024-01-01", periods=len(close), freq="D", tz="America/New_York"))


class RSIBacktestTests(unittest.TestCase):
    def test_wilder_sma_seed_and_flat_zero_loss_cases(self):
        result = rsi_indicators(bars([10, 11, 10, 12, 10]), length=2)
        self.assertTrue(math.isnan(result.rsi.iloc[1]))
        self.assertEqual(result.rsi.iloc[2], 50)  # Average gain/loss both seed at 0.5.
        self.assertAlmostEqual(result.rsi.iloc[3], 100 * 1.25 / 1.5)  # Gain=1.25, loss=.25.
        self.assertAlmostEqual(result.rsi.iloc[4], 100 * .625 / 1.75)  # Gain=.625, loss=1.125.
        self.assertEqual(rsi_indicators(bars([4] * 5), 2).rsi.iloc[-1], 50)
        self.assertEqual(rsi_indicators(bars([4, 5, 6, 7]), 2).rsi.iloc[-1], 100)

    def test_classic_wilder_14_period_seed_example(self):
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
        self.assertAlmostEqual(rsi_indicators(bars(closes), 14).rsi.iloc[-1], 70.464135, places=6)

    def test_crosses_signal_once_and_fill_at_next_open(self):
        frame = bars([10, 9, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 8, 9, 10])
        result, trades, _ = run_rsi_policy(frame, 1000, 0, 0, frame.index[0], length=2,
                                            buy_level=50, sell_level=60)
        buys = [trade for trade in trades if trade["Side"] == "Buy"]
        sells = [trade for trade in trades if trade["Side"] == "Sell"]
        self.assertEqual(len(buys), 2)  # One entry per confirmed recovery cycle, no repeated buys while above 50.
        self.assertEqual(len(sells), 1)
        self.assertEqual(buys[0]["Signal time"], frame.index[3])
        self.assertEqual(buys[0]["Fill time"], frame.index[4])
        self.assertEqual(sells[0]["Signal time"], frame.index[7])
        self.assertEqual(sells[0]["Fill time"], frame.index[8])
        self.assertFalse(result.invested.loc[frame.index[8]])
        self.assertEqual((buys[1]["Signal time"], buys[1]["Fill time"]), (frame.index[12], frame.index[13]))

    def test_prefix_invariance_and_threshold_validation(self):
        frame = bars([10, 8, 9, 7, 8, 10, 9, 12, 11, 8, 9])
        prefix = rsi_indicators(frame.iloc[:8], 3)
        full = rsi_indicators(frame, 3).iloc[:8]
        pd.testing.assert_frame_equal(prefix[["rsi", "rsi_buy", "rsi_sell"]],
                                      full[["rsi", "rsi_buy", "rsi_sell"]])
        with self.assertRaises(ValueError):
            rsi_indicators(frame, 14, 70, 30)

    def test_app_exposes_rsi_and_preserves_dates_and_interval(self):
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"
        app = AppTest.from_file(str(app_path)).run()
        self.assertFalse(app.exception)
        self.assertIn("Daily RSI recovery (buy 30 / sell 70)", app.selectbox(key="policy").options)
        start, end = app.date_input(key="date_start").value, app.date_input(key="date_end").value
        app.selectbox(key="policy").set_value("RSI recovery (14, 30/70)").run()
        self.assertEqual(app.selectbox(key="bar_interval").value, "Daily")
        self.assertEqual(app.number_input(key="rsi_length").value, 14)
        app.selectbox(key="bar_interval").set_value(5).run()
        self.assertEqual(app.selectbox(key="policy").value, "MACD turning up below zero")
        self.assertEqual((app.date_input(key="date_start").value, app.date_input(key="date_end").value),
                         (start, end))

    def test_app_runs_rsi_policy(self):
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"
        dates = pd.bdate_range("2024-01-01", "2026-10-30", tz="America/New_York")
        closes = [100 + 20 * math.sin(i / 7) for i in range(len(dates))]
        frame = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=dates)
        with patch("macd_backtest_core.fetch_daily_bars", return_value=frame), \
             patch("macd_backtest_core.fetch_corporate_actions", return_value={}):
            app = AppTest.from_file(str(app_path)).run()
            app.selectbox(key="policy").set_value("RSI recovery (14, 30/70)").run()
            app.date_input(key="date_start").set_value(pd.Timestamp("2025-10-27").date())
            app.date_input(key="date_end").set_value(pd.Timestamp("2026-10-26").date())
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("RSI, signals, and fills" in heading.value for heading in app.subheader))
        self.assertTrue(any("Final strategy value" in metric.label for metric in app.metric))


if __name__ == "__main__":
    unittest.main()
