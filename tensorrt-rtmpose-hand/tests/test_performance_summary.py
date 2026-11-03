# 检查性能统计保留离群值、正确计算分位数，并拒绝重复或缺失迭代。
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_performance import summarize_values, validate_iterations


class PerformanceTests(unittest.TestCase):
    def test_percentiles_keep_outlier(self):
        value = summarize_values([1., 2., 3., 100.])
        self.assertEqual(value['count'], 4)
        self.assertEqual(value['p50_ms'], 2.5)
        self.assertAlmostEqual(value['p95_ms'], 85.45)
        self.assertEqual(value['max_ms'], 100.)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            summarize_values([1., float('nan')])

    def test_missing_or_duplicate_iteration_rejected(self):
        validate_iterations([0, 1, 2], 3)
        for values in [[0, 1], [0, 1, 1], [0, 1, 3]]:
            with self.assertRaises(ValueError):
                validate_iterations(values, 3)


if __name__ == '__main__':
    unittest.main()
