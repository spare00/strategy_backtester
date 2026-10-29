import unittest

import pandas as pd
from pathlib import Path
from streamlit.testing.v1 import AppTest

from macd_backtest_core import NY, run_relative_momentum_policy


class RelativeMomentumPolicyTests(unittest.TestCase):
    def test_app_lists_policy_and_keeps_daily_interval(self):
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"
        app = AppTest.from_file(str(app_path)).run()
        self.assertFalse(app.exception)
        self.assertIn("Monthly ETF relative momentum rotation", app.selectbox(key="policy").options)
        app.selectbox(key="policy").set_value("ETF relative momentum rotation").run()
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


if __name__ == "__main__":
    unittest.main()
