import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('regression', Path(__file__).parents[1] / 'regression_shared_runtime.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class RegressionTests(unittest.TestCase):
    def test_changed_byte_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'
            a.write_bytes(b'1234'); b.write_bytes(b'1234')
            m.require_equal(a, b)
            b.write_bytes(b'1235')
            with self.assertRaises(RuntimeError): m.require_equal(a, b)
    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError): m.new_directory(Path(d))

if __name__ == '__main__': unittest.main()
