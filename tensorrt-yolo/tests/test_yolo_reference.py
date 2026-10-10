"""Regression checks for the fixed640 reference contract."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_yolo_reference.py"


class ReferenceTests(unittest.TestCase):
    def module(self):
        self.assertTrue(SCRIPT.is_file(), "Fixed640 reference script is missing")
        spec = importlib.util.spec_from_file_location("reference", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_camera_transpose_not_rotation(self):
        module = self.module()
        frame = np.zeros((360, 640, 3), np.uint8)
        frame[10, 20] = [7, 19, 251]
        result = module.resize_business_frame(frame)
        self.assertEqual(result.shape, (640, 360, 3))
        np.testing.assert_array_equal(result[20, 10], [7, 19, 251])

    def test_mean_mask_recomputes_current_image_and_truncates(self):
        module = self.module()
        rgb = np.zeros((2, 2, 3), np.uint8)
        rgb[0, 0] = [100, 103, 107]
        first = module.mean_region_mask(rgb, np.array([[0, 0]] * 3, np.int32))
        second = module.mean_region_mask(first, np.array([[1, 1]] * 3, np.int32))
        np.testing.assert_array_equal(first[0, 0], [25, 25, 26])
        np.testing.assert_array_equal(second[1, 1], [6, 6, 6])
        np.testing.assert_array_equal(rgb[0, 0], [100, 103, 107])

    def test_fixed_padding_rgb_and_binary_layout(self):
        module = self.module()
        frame = np.full((640, 360, 3), [10, 20, 30], np.uint8)
        result, ratio, padding = module.letterbox_fixed(frame)
        self.assertEqual(ratio, 1.0)
        self.assertEqual(padding, (140, 0, 140, 0))
        tensor = module.to_tensor(result)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.dtype("float32"))
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_array_equal(tensor[0, :, 0, 140], np.array([30, 20, 10], np.float32) / 255)
        np.testing.assert_array_equal(result[:, :140], 114)
        np.testing.assert_array_equal(result[:, 500:], 114)

    def test_real_polygons_and_order_match_historical_mask(self):
        # Digest generated independently with v21's original MeanRegionMask,
        # RGB PIL roundtrip, headrest then window, on this nonuniform fixture.
        module = self.module()
        y, x = np.indices((640, 360))
        pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)
        result = module.apply_business_masks(pixels)
        self.assertEqual(hashlib.sha256(result.tobytes()).hexdigest(),
                         "654839fe9583502bb8b9e2695072e852ed8a9b798a0e773c9de1909dc6bcabc5")

    def test_odd_padding_rounding(self):
        module = self.module()
        result, ratio, padding = module.letterbox_fixed(np.zeros((640, 359, 3), np.uint8))
        self.assertEqual(padding, (140, 0, 141, 0))
        self.assertEqual(result.shape, (640, 640, 3))

    def test_business_cli_does_not_transpose_and_bin_matches_npy(self):
        self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.png", root / "reference"
            frame = np.zeros((640, 360, 3), np.uint8)
            frame[500, 200] = [10, 20, 30]
            self.assertTrue(cv2.imwrite(str(source), frame))
            command = [sys.executable, str(SCRIPT), "--input", str(source), "--input-stage", "business", "--output-dir", str(output)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            np.testing.assert_array_equal(cv2.imread(str(output / "business.png")), frame)
            tensor = np.load(output / "input.npy", allow_pickle=False)
            binary = np.fromfile(output / "input.bin", dtype="<f4").reshape(1, 3, 640, 640)
            np.testing.assert_array_equal(tensor, binary)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["input_stage"], "business")
            self.assertEqual(metadata["letterbox"]["padding_ltrb"], [140, 0, 140, 0])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_wrong_business_geometry_is_rejected(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module.prepare_reference(np.zeros((360, 640, 3), np.uint8), "business")


if __name__ == "__main__":
    unittest.main()
