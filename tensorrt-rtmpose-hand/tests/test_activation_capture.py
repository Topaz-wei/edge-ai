# 验证采集记录拒绝 engine 替换，且非有限 SimCC 不会让报告生成中断。
import json
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from capture_activation_probe import verify_capture
from activation_metrics import safe_final_comparison

class CaptureTests(unittest.TestCase):
    def test_engine_swap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            engine = root / 'engine'
            engine.write_bytes(b'changed')
            (root / 'capture.json').write_text(json.dumps({'assets_sha256': {str(engine): 'wrong'}, 'raw_sha256': {}}))
            with self.assertRaises(ValueError):
                verify_capture(root)

    def test_nonfinite_final_reported(self):
        raw = [np.full((1, 21, 512), np.nan), np.zeros((1, 21, 512))]
        result = safe_final_comparison(raw, raw, [])
        self.assertEqual(result['status'], 'nonfinite')
        self.assertEqual(result['actual_nonfinite'], 21 * 512)

if __name__ == '__main__':
    unittest.main()
