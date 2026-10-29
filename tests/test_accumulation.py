import math
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from macd_backtest_core import NY, accumulation_indicators, run_accumulation_policy


POLICY = "Accumulation breakout"


def synthetic_bars(count=95):
    close = [100.4] * count
    volume = [100.0] * count
    close[80], volume[80] = 101.0, 200.0
    close[81] = 102.0
    close[82:] = [99.0] * (count - 82)
    frame = pd.DataFrame({"open": close, "high": [value + 0.1 for value in close],
                          "low": [value - 0.4 for value in close], "close": close,
                          "volume": volume},
                         index=pd.bdate_range("2023-01-02", periods=count, tz=NY))
    return frame


class AccumulationTests(unittest.TestCase):
    def test_tight_base_positive_money_flow_breakout_and_exit(self):
        frame = synthetic_bars()
        indicators = accumulation_indicators(frame)
        self.assertTrue(indicators.accumulation_buy.iloc[80])
        self.assertGreaterEqual(indicators.accum_cmf20.iloc[80], 0.05)
        self.assertGreaterEqual(indicators.accum_volume_ratio.iloc[80], 1.5)
        result, trades, _ = run_accumulation_policy(
            frame, 10_000, 0, 0, frame.index[60],
        )
        self.assertEqual([trade["Side"] for trade in trades], ["Buy", "Sell"])
        self.assertEqual(trades[0]["Signal time"], frame.index[80])
        self.assertEqual(trades[0]["Fill time"], frame.index[81])
        self.assertEqual(trades[1]["Signal time"], frame.index[82])
        self.assertEqual(trades[1]["Fill time"], frame.index[83])
        self.assertFalse(result.invested.loc[frame.index[83]])

    def test_indicator_is_prefix_invariant_and_validates_inputs(self):
        frame = synthetic_bars()
        prefix = accumulation_indicators(frame.iloc[:80])
        full = accumulation_indicators(frame).iloc[:80]
        columns = ["base_high", "base_low", "base_range_pct", "accum_cmf20",
                   "accum_volume_ratio", "accumulation_buy", "accumulation_sell"]
        pd.testing.assert_frame_equal(prefix[columns], full[columns])
        with self.assertRaises(ValueError):
            accumulation_indicators(frame, base_days=10)

    def test_app_exposes_and_runs_accumulation_policy(self):
        app_path = Path(__file__).resolve().parents[1] / "macd_backtest_app.py"
        frame = synthetic_bars(650)
        with patch("macd_backtest_core.fetch_daily_bars", return_value=frame), \
             patch("macd_backtest_core.fetch_corporate_actions", return_value={}):
            app = AppTest.from_file(str(app_path)).run()
            self.assertIn("Accumulation breakout (base + volume confirmation)",
                          app.selectbox(key="policy").options)
            app.selectbox(key="policy").set_value(POLICY).run()
            self.assertEqual(app.selectbox(key="bar_interval").value, "Daily")
            app.date_input(key="date_start").set_value(frame.index[400].date())
            app.date_input(key="date_end").set_value(frame.index[-1].date())
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Accumulation proxy (CMF), signals, and fills" in heading.value
                            for heading in app.subheader))


if __name__ == "__main__":
    unittest.main()
