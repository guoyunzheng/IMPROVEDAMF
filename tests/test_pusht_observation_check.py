"""Geometry checks with known poses; no simulator or demonstration data needed."""

import unittest

import numpy as np

from scripts.check_pusht_observations import compare_keypoints


class ObservationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.states = np.array([[0, 0, 100, 200, 0], [0, 0, 300, 400, np.pi / 2]])
        self.local = np.array([[1, 0], [0, 2], [-3, -1]])
        self.recorded = np.array([[[101, 200], [100, 202], [97, 199]],
                                  [[300, 401], [298, 400], [301, 397]]])

    def test_translation_and_quarter_turn_preserve_identity(self):
        result = compare_keypoints(self.states, self.recorded, self.local)
        self.assertLess(result["ordered_keypoint_error_px_max"], 1e-10)
        self.assertLess(result["recorded_local_scatter_px_max"], 1e-10)
        self.assertEqual(result["dataset_to_environment_keypoint_index"], [0, 1, 2])

    def test_point_permutation_is_distinguished_from_geometry_change(self):
        result = compare_keypoints(self.states, self.recorded, self.local[[2, 0, 1]])
        self.assertGreater(result["ordered_keypoint_error_px_mean"], 1)
        self.assertLess(result["after_best_permutation_error_px_max"], 1e-10)
        self.assertEqual(result["dataset_to_environment_keypoint_index"], [1, 2, 0])

    def test_geometry_difference_remains_after_permutation(self):
        result = compare_keypoints(self.states, self.recorded, self.local + 100)
        self.assertGreater(result["after_best_permutation_error_px_mean"], 100)

    def test_recording_pose_inconsistency_is_detected(self):
        recorded = self.recorded.copy()
        recorded[1, 0, 0] += 20
        result = compare_keypoints(self.states, recorded, self.local)
        self.assertGreater(result["recorded_local_scatter_px_max"], 5)


if __name__ == "__main__":
    unittest.main()
