import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_validation_annotations import yolo_line


class LabelTests(unittest.TestCase):
    def test_normalizes_original_coordinates(self):
        self.assertEqual(
            yolo_line([0, 0, 360, 640], 360, 640),
            "0 0.50000000 0.50000000 1.00000000 1.00000000",
        )
        self.assertEqual(
            yolo_line([90, 160, 270, 480], 360, 640),
            "0 0.50000000 0.50000000 0.50000000 0.50000000",
        )

    def test_rejects_invalid_boxes(self):
        for b in (
            [10, 0, 0, 10],
            [-1, 0, 10, 10],
            [0, 0, 361, 640],
            [0, 0, float("nan"), 10],
        ):
            with self.assertRaises(ValueError):
                yolo_line(b, 360, 640)


if __name__ == "__main__":
    unittest.main()
