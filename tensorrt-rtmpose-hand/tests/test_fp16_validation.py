# 本测试验证 FP16 比较不丢低分点、完成记录绑定，以及 inspector 未知信息处理。
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(
        Path(
            __file__
        ).resolve().parents[1] / 'scripts'
    )
)
import verify_trt_fp16 as verify


class ValidationTests(unittest.TestCase):
    def test_low_score_point_is_included(self):
        x = np.zeros(
            (1, 21, 512),
            dtype=np.float32
        )
        y = x.copy()
        x[:, :, 0] = y[:, :, 0] = 0.0001
        changed = x.copy()
        changed[0, 0, 0] = 0
        changed[0, 0, 2] = 0.0001
        samples = [{'bbox': [0, 0, 100, 100]}]
        result = verify.compare_raw(
            (changed, y),
            (x, y),
            samples
        )
        self.assertEqual(
            result['input_coordinate_max_l2_px'],
            1.0
        )
        self.assertAlmostEqual(
            result['input_coordinate_mean_l2_px'],
            1 / 21
        )
        self.assertEqual(
            result['valid_mask_flip_count'],
            0
        )

    def test_gates_reject_coordinate_and_score_changes(self):
        raw = np.zeros(
            (1, 21, 512),
            dtype=np.float32
        )
        raw[:, :, 0] = 1.0
        metrics = verify.compare_raw(
            (raw, raw),
            (raw, raw),
            [{'bbox': [0, 0, 100, 100]}]
        )
        limits = verify.read_json(
            verify.CONFIG
        )['acceptance']
        self.assertEqual(
            verify.gate_failures(
                metrics,
                limits
            ),
            []
        )
        metrics['input_coordinate_max_l2_px'] = limits['input_coordinate_max_l2_px'] + 0.5
        metrics['valid_mask_flip_count'] = 1
        failures = verify.gate_failures(
            metrics,
            limits
        )
        self.assertIn(
            'input_coordinate_max_l2_px',
            failures
        )
        self.assertIn(
            'valid_mask_flip_count',
            failures
        )

    def test_tampered_lock_acceptance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            limits = dict(verify.read_json(verify.CONFIG)['acceptance'])
            limits['score_max_abs'] = 100.0
            verify.write_json(
                folder / 'input-lock.json',
                {'assets_sha256': {str(verify.CONFIG): verify.digest(verify.CONFIG)},
                 'acceptance': limits}
            )
            with patch.object(verify, 'OUTPUT', folder):
                with self.assertRaisesRegex(AssertionError, 'acceptance'):
                    verify.compare()

    def test_nonfinite_is_rejected(self):
        raw = np.zeros(
            (1, 21, 512),
            dtype=np.float32
        )
        raw[0, 0, 0] = np.nan
        with self.assertRaises(
            AssertionError
        ):
            verify.compare_raw(
                (raw, raw),
                (raw, raw),
                [{'bbox': [0, 0, 1, 1]}]
            )

    def test_inspector_does_not_infer_unknown_precision(self):
        layers = [
            {'Name': 'unknown', 'LayerType': 'Convolution', 'Inputs': [],
             'Outputs': [{'Format/Datatype': 'N/A due to dynamic shapes'}]},
            {'Name': 'fp16', 'LayerType': 'Convolution', 'Inputs': [],
             'Outputs': [{'Format/Datatype': 'Row major linear FP16'}]},
            {'Name': 'convert', 'LayerType': 'Reformat',
             'Inputs': [{'Format/Datatype': 'Row major linear FP16'}],
             'Outputs': [{'Format/Datatype': 'Row major linear FP32'}]}
        ]
        result = verify.summarize_layers(
            {'Layers': layers}
        )
        self.assertEqual(
            result['io_precision_counts'],
            {'unknown': 1, 'fp16': 1, 'mixed_fp16_fp32': 1}
        )
        self.assertEqual(
            result['reformat_layers'],
            ['convert']
        )

    def test_build_completion_rejects_replaced_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            engine = folder / 'engine'
            engine.write_bytes(b'engine-a')
            for name in ['build-provenance.json', 'build.log']:
                (folder / name).write_text('{}')
            completion = {
                'engine_sha256': verify.digest(engine),
                'provenance_sha256': verify.digest(folder / 'build-provenance.json'),
                'log_sha256': verify.digest(folder / 'build.log'),
            }
            verify.write_json(
                folder / 'build-complete.json',
                completion
            )
            verify.validate_build_completion(
                folder,
                engine
            )
            engine.write_bytes(b'engine-b')
            with self.assertRaises(AssertionError):
                verify.validate_build_completion(
                    folder,
                    engine
                )

    def test_completion_rejects_stale_engine_and_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(
                temporary
            )
            payload = folder / 'output.bin'
            payload.write_bytes(
                b'raw'
            )
            completion = {'engine_sha256': 'engine', 'cases_sha256': 'cases',
                          'mode': 'npy', 'samples': 1,
                          'files': [{'path': 'output.bin', 'sha256': verify.digest(
                              payload
                          )}]}
            verify.write_json(
                folder / 'complete.json',
                completion
            )
            verify.validate_completion(
                folder,
                'engine',
                'cases',
                'npy',
                1,
                {'output.bin'}
            )
            with self.assertRaises(
                AssertionError
            ):
                verify.validate_completion(
                    folder,
                    'other-engine',
                    'cases',
                    'npy',
                    1,
                    {'output.bin'}
                )
            with self.assertRaises(
                AssertionError
            ):
                verify.validate_completion(
                    folder,
                    'engine',
                    'cases',
                    'npy',
                    1,
                    {'missing.bin'}
                )
            payload.write_bytes(
                b'changed'
            )
            with self.assertRaises(
                AssertionError
            ):
                verify.validate_completion(
                    folder,
                    'engine',
                    'cases',
                    'npy',
                    1,
                    {'output.bin'}
                )


if __name__ == '__main__':
    unittest.main()
