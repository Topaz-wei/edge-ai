# 验证并列峰按首个 bin 解码，以及微小扰动导致远峰切换的诊断数据。
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_simcc_margin import axis_metrics


class MarginTests(unittest.TestCase):
    def test_distant_peak_switch(self):
        reference = np.array([0, .801, 0, 0, .799], dtype=np.float32)
        actual = np.array([0, .798, 0, 0, .803], dtype=np.float32)
        result = axis_metrics(reference, actual)
        self.assertEqual(result['reference_top1_bin'], 1)
        self.assertEqual(result['actual_top1_bin'], 4)
        self.assertEqual(result['shift_bins'], 3)
        self.assertAlmostEqual(result['reference_winner_gap'], .002, places=6)
        self.assertAlmostEqual(result['actual_winner_gap'], .005, places=6)

    def test_ties_choose_first_bin(self):
        values = np.array([0, 1, 1, 0], dtype=np.float32)
        result = axis_metrics(values, values)
        self.assertEqual(result['reference_top1_bin'], 1)
        self.assertEqual(result['reference_top2_bin'], 2)
        self.assertEqual(result['reference_margin'], 0)
        self.assertEqual(result['shift_bins'], 0)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            axis_metrics(np.array([0, np.nan]), np.array([0, 1]))


if __name__ == '__main__':
    unittest.main()
