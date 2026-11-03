# 真实固定数据的CPU负面测试；只复制小清单，模型和张量只读引用，不修改原数据。
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'tensorrt-rtmpose-hand'
BINARY = ROOT / 'build-shared/tensorrt-rtmpose-hand/rtmpose_int8'


class AuditTests(unittest.TestCase):
    def run_bad_order(self, change):
        with tempfile.TemporaryDirectory() as temporary:
            module = Path(temporary)
            for name in ['models', 'configs', 'scripts', 'src']:
                (module / name).symlink_to(MODULE / name)
            calibration = module / 'calibration'
            calibration.mkdir()
            for name in ['hand_boxes.jsonl', 'split.json', 'development_tensor_order.txt', 'coverage.json']:
                shutil.copyfile(MODULE / 'calibration' / name, calibration / name)
            entries = (MODULE / 'calibration/calibration_tensor_order.txt').read_text().splitlines()
            change(entries)
            (calibration / 'calibration_tensor_order.txt').write_text('\n'.join(entries) + '\n')
            result = subprocess.run(
                [str(BINARY), 'audit', str(module), str(module / 'audit.json')],
                capture_output=True,
                text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((module / 'audit.json').exists())
            return result.stderr

    def test_duplicate_tensor_rejected(self):
        message = self.run_bad_order(lambda rows: rows.__setitem__(1, rows[0]))
        self.assertIn('duplicate', message)

    def test_development_injection_rejected(self):
        dev = (MODULE / 'calibration/development_tensor_order.txt').read_text().splitlines()[0]
        message = self.run_bad_order(lambda rows: rows.__setitem__(0, dev))
        self.assertIn('coverage identity mismatch', message)

    def test_reordered_tensor_list_rejected(self):
        message = self.run_bad_order(lambda rows: rows.reverse())
        self.assertIn('coverage identity mismatch', message)


if __name__ == '__main__':
    unittest.main()
