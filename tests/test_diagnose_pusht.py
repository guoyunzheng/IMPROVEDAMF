"""CPU regression checks for checkpoint inspection; no dataset is required."""

import unittest

import torch

from scripts.diagnose_pusht import classify_nonfinite_state


class CheckpointInspectionTests(unittest.TestCase):
    def test_standard_causal_attention_masks_are_expected(self):
        state = {
            "model.mask": torch.nn.Transformer.generate_square_subsequent_mask(4),
            "model.memory_mask": torch.tensor([
                [0.0, -float("inf"), -float("inf")],
                [0.0, 0.0, -float("inf")],
            ]),
            "model.projection.weight": torch.ones(2, 3),
        }
        unexpected, masks = classify_nonfinite_state(state)
        self.assertEqual(unexpected, [])
        self.assertEqual(masks, ["model.mask", "model.memory_mask"])

    def test_nan_and_positive_infinity_in_masks_are_rejected(self):
        for key in ("model.mask", "model.memory_mask"):
            for invalid in (float("nan"), float("inf")):
                with self.subTest(key=key, invalid=invalid):
                    state = {key: torch.tensor([[0.0, invalid]])}
                    self.assertEqual(classify_nonfinite_state(state), ([key], []))

    def test_fully_blocked_attention_row_is_rejected(self):
        key = "model.mask"
        state = {key: torch.tensor([[0.0, -float("inf")],
                                   [-float("inf"), -float("inf")]])}
        self.assertEqual(classify_nonfinite_state(state), ([key], []))

    def test_nonfinite_weights_and_unknown_buffers_are_rejected(self):
        for key in ("model.projection.weight", "normalizer.params_dict.action.scale",
                    "model.unknown_mask"):
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, invalid=invalid):
                    state = {key: torch.tensor([[0.0, invalid]])}
                    self.assertEqual(classify_nonfinite_state(state), ([key], []))

    def test_finite_state_passes(self):
        state = {"model.mask": torch.zeros(2, 2), "weight": torch.ones(3),
                 "count": torch.tensor(10), "metadata": "example"}
        self.assertEqual(classify_nonfinite_state(state), ([], []))


if __name__ == "__main__":
    unittest.main()
