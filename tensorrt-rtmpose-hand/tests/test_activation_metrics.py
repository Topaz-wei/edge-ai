# 激活误差统计的边界测试：整体缩放、零范数、非有限值与形状不匹配。
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from activation_metrics import metrics

class MetricsTests(unittest.TestCase):
    def test_scale_has_cosine_one(self):
        value = metrics(np.array([1., 2.]), np.array([2., 4.]))
        self.assertAlmostEqual(value['cosine'], 1.)
        self.assertAlmostEqual(value['relative_l2'], 1.)
        self.assertAlmostEqual(value['norm_ratio'], 2.)

    def test_zero_norm(self):
        value = metrics(np.zeros(2), np.zeros(2))
        self.assertIsNone(value['cosine'])
        self.assertIsNone(value['relative_l2'])
        self.assertEqual(value['zero_norm_case'], 'both')
        self.assertEqual(metrics(np.ones(2), np.zeros(2))['zero_norm_case'], 'actual')

    def test_nonfinite(self):
        value = metrics(np.ones(2), np.array([1., np.inf]))
        self.assertEqual(value['actual_nonfinite'], 1)
        self.assertIsNone(value['cosine'])

    def test_shape(self):
        with self.assertRaises(ValueError):
            metrics(np.ones(2), np.ones((1, 2)))

if __name__ == '__main__':
    unittest.main()
