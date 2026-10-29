import unittest

import pandas as pd

from macd_backtest_core import NY, run_daily_policy


class MonthlyTrendPolicyTests(unittest.TestCase):
    def test_month_end_signal_executes_on_next_month_session(self):
        dates = pd.bdate_range("2023-01-01", "2024-01-31", tz=NY)
        month_values = {month: 100.0 for month in range(1, 10)}
        month_values.update({10: 110.0, 11: 90.0, 12: 100.0, 1: 100.0})
        close = [month_values[stamp.month] for stamp in dates]
        frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=dates)

        result, trades, _ = run_daily_policy(
            frame, 10_000, "Monthly 10-month SMA risk filter", 0, 0, dates[0])

        self.assertEqual([trade["Side"] for trade in trades], ["Buy", "Sell"])
        self.assertEqual([trade["Fill time"] for trade in trades], [pd.Timestamp("2023-11-01", tz=NY),
                                                                    pd.Timestamp("2023-12-01", tz=NY)])
        self.assertIn("monthly_sma10", result)


if __name__ == "__main__":
    unittest.main()
