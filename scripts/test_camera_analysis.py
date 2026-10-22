"""Regression checks for comparable service time and long-tail statistics."""
import unittest
from analyze_camera_architecture import distribution, model_service_ms, long_interval, validate_read_timing


class CameraAnalysisTests(unittest.TestCase):
    def test_read_split_rejects_inconsistent_evidence(self):
        valid = {'frame': 2, 'read_ms': 30, 'grab_ms': 22, 'retrieve_ms': 8}
        validate_read_timing(valid)
        validate_read_timing({'frame': 2, 'read_ms': 30})  # historical CSV
        validate_read_timing({'frame': 1, 'read_ms': 0, 'grab_ms': 0, 'retrieve_ms': 0})
        validate_read_timing(dict(valid, read_ms=30.000001))  # CSV rounding
        for changes in ({'read_ms': 31}, {'grab_ms': -1, 'retrieve_ms': 31},
                        {'frame': 1}, {'grab_ms': float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_read_timing(dict(valid, **changes))
        with self.assertRaises(ValueError):
            validate_read_timing({'frame': 2, 'read_ms': 30, 'grab_ms': 22})

    def test_p99_preserves_rare_tail(self):
        result = distribution([1, 2, 3, 4, 100])
        self.assertAlmostEqual(result['p99'], 96.16)
        self.assertEqual(result['max'], 100)

    def test_compute_includes_moved_cpu_preparation(self):
        self.assertEqual(model_service_ms({'geometry_ms': 2, 'pipeline_ms': 10}), 12)
        self.assertEqual(model_service_ms({'geometry_ms': 2, 'prepare_ms': 5, 'pipeline_ms': 10}), 17)

    def test_tail_preserves_queue_and_log_boundary_evidence(self):
        before = {'frame': 30, 'end_ms': 100}
        after = {'frame': 31, 'start_ms': 95, 'end_ms': 1110, 'total_ms': 1015,
                 'read_ms': 4, 'geometry_ms': 0, 'prepare_ms': 3, 'pipeline_ms': 10,
                 'display_ms': 0, 'draw_ms': 0, 'encode_ms': 0, 'queue_wait_ms': 998}
        result = long_interval(before, after)
        self.assertEqual(result['interval_ms'], 1010)
        self.assertEqual(result['prepare_ms'], 3)
        self.assertEqual(result['queue_wait_ms'], 998)
        self.assertTrue(result['after_logged_frame'])
        self.assertEqual(result['start_minus_previous_end_ms'], -5)
        self.assertEqual(result['estimated_consumer_gap_ms'], 1000)


if __name__ == '__main__':
    unittest.main()
