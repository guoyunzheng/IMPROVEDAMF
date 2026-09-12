"""Run with `python -m unittest discover -s tests` in the MetaWorld environment."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr

from scripts.generate_metaworld_demos import generate_demos, normalize_task


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.closed = False

    def seed(self, seed):
        self.current_seed = seed

    def reset(self, seed):
        assert seed == self.current_seed
        return np.arange(39, dtype=np.float32), {}

    def step(self, action):
        # The first seed fails; subsequent seeds succeed in one step.
        return np.zeros(39), 0.0, False, True, {"success": self.current_seed > 0}

    def close(self):
        self.closed = True


class DemoGenerationTests(unittest.TestCase):
    def test_task_names(self):
        for name in ("pick-out-of-hole", "pick-out-of-hole-v3", "pick_out_of_hole_v3"):
            self.assertEqual(normalize_task(name), "pick-out-of-hole")

    def test_successful_episodes_and_observation_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo.zarr"
            env = FakeEnv()
            with patch("scripts.generate_metaworld_demos.gym.make", return_value=env), \
                 patch("scripts.generate_metaworld_demos.get_policy") as policy:
                policy.return_value.get_action.return_value = np.ones(4)
                generate_demos("reach_v3", num_demos=2, output=output, max_attempts=3)
            data = zarr.open_group(str(output), mode="r")
            np.testing.assert_array_equal(data["meta/episode_ends"][:], [1, 2])
            np.testing.assert_array_equal(data["data/obs"][0], np.r_[np.arange(18), [36, 37, 38]])
            np.testing.assert_array_equal(data["data/action"][:], np.ones((2, 4)))
            self.assertEqual(data.attrs["successful_seeds"], [1, 2])
            self.assertTrue(data.attrs["complete"])
            self.assertTrue(env.closed)

    def test_attempt_limit_closes_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo.zarr"
            env = FakeEnv()
            with patch("scripts.generate_metaworld_demos.gym.make", return_value=env), \
                 patch("scripts.generate_metaworld_demos.get_policy") as policy:
                policy.return_value.get_action.return_value = np.ones(4)
                with self.assertRaisesRegex(RuntimeError, "Collected 0/1"):
                    generate_demos("reach", num_demos=1, output=output, max_attempts=1)
            self.assertTrue(env.closed)
            data = zarr.open_group(str(output), mode="r")
            self.assertFalse(data.attrs["complete"])
            self.assertEqual(data["meta/episode_ends"].shape, (0,))

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo.zarr"
            output.write_text("existing data")
            with self.assertRaises(FileExistsError):
                generate_demos("reach", output=output)
            self.assertEqual(output.read_text(), "existing data")


if __name__ == "__main__":
    unittest.main()
