# 验证线性层误差拆分的恒等式，避免把上游误差传播误判为该层自身误差。
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_projection_error import decompose

class ProjectionTests(unittest.TestCase):
    def test_upstream_only(self):
        weight = np.eye(2)
        x = np.array([[1., 2.]])
        z = np.array([[1.1, 2.2]])
        result = decompose(x, z, x, z, weight)
        self.assertAlmostEqual(result['execution_residual_l2'], 0.)
        self.assertAlmostEqual(result['total_l2'], result['propagated_l2'])

    def test_local_only(self):
        x = np.array([[1., 2.]])
        result = decompose(x, x, x, x + 1, np.eye(2))
        self.assertAlmostEqual(result['propagated_l2'], 0.)
        self.assertAlmostEqual(result['execution_residual_l2'], np.sqrt(2.))
        self.assertLess(result['closure_max_abs'], 1e-12)

if __name__ == '__main__':
    unittest.main()
