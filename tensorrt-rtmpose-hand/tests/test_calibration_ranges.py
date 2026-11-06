# 验证 cache IEEE754 解码、非有限条目保留与非法输入拒绝，避免把异常 scale 静默丢弃。
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_calibration_ranges import read_scales


class CacheRangeTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache'
            path.write_text(content)
            return read_scales(path)

    def test_scale_and_infinity_preserved(self):
        rows = self.parse('TRT-8502-EntropyCalibration2\ninput: 3f800000\nmask: 7f800000\n')
        self.assertEqual(rows[0]['scale'], 1)
        self.assertEqual(rows[0]['symmetric_threshold_127s'], 127)
        self.assertEqual(rows[0]['int8_grid_min'], -128)
        self.assertTrue(math.isinf(rows[1]['scale']))

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            self.parse('TRT-8502-EntropyCalibration2\nx: 3f800000\nx: 3f800000\n')

    def test_wrong_header_rejected(self):
        with self.assertRaises(ValueError):
            self.parse('TRT-other\nx: 3f800000\n')


if __name__ == '__main__':
    unittest.main()
