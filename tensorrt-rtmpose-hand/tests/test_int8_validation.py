# 验证INT8统计不把混合层算成纯INT8，且RMSE和argmax统计保留所有点。
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_trt_int8 import axis_statistics, layer_summary, ordered_tensors


class Int8Tests(unittest.TestCase):
    def test_rmse_and_argmax_shift(self):
        reference = np.array([[[1., 0., 0.]]], dtype=np.float32)
        actual = np.array([[[0., 0., 2.]]], dtype=np.float32)
        value = axis_statistics(actual, reference)
        self.assertAlmostEqual(value['rmse'], np.sqrt(5. / 3.))
        self.assertEqual(value['argmax_changed_count'], 1)
        self.assertEqual(value['argmax_shift_max_abs_bins'], 2)

    def test_int8_and_mixed_classification(self):
        layers = {'Layers': [
            {'Name': 'conv', 'LayerType': 'Convolution', 'Inputs': [{'Format/Datatype': 'INT8'}], 'Outputs': [{'Name': 'hidden', 'Format/Datatype': 'INT8'}]},
            {'Name': 'output', 'LayerType': 'Reformat', 'Inputs': [{'Name': 'hidden', 'Format/Datatype': 'INT8'}], 'Outputs': [{'Name': 'simcc_x', 'Format/Datatype': 'FP32'}]},
            {'Name': 'shape', 'Inputs': [], 'Outputs': []},
        ]}
        value = layer_summary(layers)
        self.assertEqual(value['io_precision_counts'], {'int8': 1, 'mixed_int8_fp32': 1, 'non_float_or_unknown': 1})
        self.assertEqual(value['fp16_tensor_layers'], 0)
        self.assertEqual(value['output_branches']['simcc_x'][0]['name'], 'output')

    def test_legacy_metadata_structure(self):
        entries = [{'path': 'example.npy', 'sha256': 'abc'}]
        fields = ['inference_profile', '1/2/4', 'workspace_mib', 2048,
                  'fp16', 0, 'tf32', 0, 'input_name', 'input',
                  'input_dtype', 'FP32', 'ordered_tensors', entries]
        self.assertEqual(ordered_tensors({'calibration_shape': fields}), entries)
        self.assertEqual(ordered_tensors({'ordered_tensors': entries}), entries)
        fields[3] = 1024
        with self.assertRaises(AssertionError):
            ordered_tensors({'calibration_shape': fields})

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            axis_statistics(np.array([np.nan]), np.array([0.]))


if __name__ == '__main__':
    unittest.main()
