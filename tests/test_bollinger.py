import unittest

import pandas as pd

from macd_backtest_core import NY, run_daily_policy


class BollingerPolicyTests(unittest.TestCase):
    def test_lower_band_entry_and_middle_band_exit_fill_next_open(self):
        close = [100.0] * 20 + [50.0, 60.0, 80.0, 100.0, 100.0, 100.0]
        index = pd.bdate_range("2024-01-02", periods=len(close), tz=NY)
        frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=index)

        result, trades, _ = run_daily_policy(
            frame, 10_000, "Bollinger Band mean reversion", 0, 0, index[0],
            bollinger_length=5, bollinger_stddev=1.8)

        self.assertEqual([trade["Side"] for trade in trades], ["Buy", "Sell"])
        self.assertEqual([trade["Fill time"] for trade in trades], [index[21], index[23]])
        self.assertIn("bb_upper", result)
        self.assertIn("bb_middle", result)
        self.assertIn("bb_lower", result)


if __name__ == "__main__":
    unittest.main()
