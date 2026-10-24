"""CPU-only checks for exact phase lengths and final-epoch persistence."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from source.common.mean_flow_schedule import scheduled_flow_ratio


class ExactPhaseTests(unittest.TestCase):
    def test_exact_400_plus_400_overrides_fraction(self):
        schedule = {"enabled": True, "fm_only_fraction": 0.8, "fm_only_epochs": 400}
        actual = [scheduled_flow_ratio(epoch, 800, 0.2, schedule) for epoch in range(800)]
        self.assertEqual(actual, [1.0] * 400 + [0.2] * 400)

    def test_preserve_previous_boundary_and_extend_mixed_phase(self):
        schedule = {"enabled": True, "fm_only_epochs": 401}
        actual = [scheduled_flow_ratio(epoch, 801, 0.2, schedule) for epoch in range(801)]
        self.assertEqual(actual, [1.0] * 401 + [0.2] * 400)

    def test_resume_uses_absolute_epoch(self):
        schedule = {"fm_only_epochs": 400}
        self.assertEqual([scheduled_flow_ratio(epoch, 800, 0.2, schedule)
                          for epoch in range(650, 800)], [0.2] * 150)

    def test_legacy_fraction_and_missing_schedule_unchanged(self):
        schedule = {"enabled": True, "fm_only_fraction": 0.8}
        self.assertEqual([scheduled_flow_ratio(epoch, 501, 0.2, schedule)
                          for epoch in range(501)], [1.0] * 401 + [0.2] * 100)
        for value in (None, {"enabled": False, "fm_only_epochs": 400}):
            self.assertEqual(scheduled_flow_ratio(0, 800, 0.5, value), 0.5)

    def test_phase_edges_and_pure_mf(self):
        self.assertEqual(scheduled_flow_ratio(0, 800, 0.2, {"fm_only_epochs": 0}), 0.2)
        self.assertEqual(scheduled_flow_ratio(799, 800, 0.2, {"fm_only_epochs": 800}), 1.0)
        self.assertEqual(scheduled_flow_ratio(400, 800, 0.0, {"fm_only_epochs": 400}), 0.0)

    def test_invalid_exact_lengths_rejected(self):
        for length in (-1, 801, 400.5, True, "400"):
            with self.subTest(length=length), self.assertRaises(ValueError):
                scheduled_flow_ratio(0, 800, 0.2, {"fm_only_epochs": length})


class TrainingConfig(dict):
    __getattr__ = dict.__getitem__


class FinalEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the workspace's actual event predicates, without importing
        # Torch/Hydra or copying their logic into an independent implementation.
        path = Path(__file__).resolve().parents[1] / "source/workspace/train_mean_flow_lowdim_workspace.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        workspace = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        run = next(node for node in workspace.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        loop = next(node for node in ast.walk(run) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "local_epoch_idx")
        final = next(node for node in loop.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "is_final_epoch" for target in node.targets))
        start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "rollout_this_epoch" for target in node.targets))
        end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Name) and node.test.id == "rollout_this_epoch")
        cls.events = compile(ast.Module(body=[final, *loop.body[start:end]], type_ignores=[]), str(path), "exec")
        cls.predicates = {}
        for key in ("val_every", "checkpoint_every"):
            branch = next(node for node in loop.body if isinstance(node, ast.If)
                          and any(isinstance(part, ast.Attribute) and part.attr == key for part in ast.walk(node.test)))
            cls.predicates[key] = compile(ast.Expression(branch.test), str(path), "eval")

    def events_at(self, epoch, total, late=True, final_n=100, final_interval=50):
        training = TrainingConfig(num_epochs=total, rollout_every=50, checkpoint_every=50, val_every=25)
        if late:
            training["last_n_rollout_every"] = SimpleNamespace(n=final_n, interval=final_interval)
        scope = {"self": SimpleNamespace(epoch=epoch), "cfg": SimpleNamespace(training=training)}
        exec(self.events, scope)
        return (scope["rollout_this_epoch"], eval(self.predicates["checkpoint_every"], scope),
                eval(self.predicates["val_every"], scope))

    def test_final_epoch_799_is_evaluated_validated_and_saved(self):
        for late in (True, False):
            with self.subTest(late=late):
                self.assertEqual(self.events_at(799, 800, late), (True, True, True))

    def test_previous_501_epoch_timing_is_unchanged(self):
        for epoch in range(501):
            self.assertEqual(self.events_at(epoch, 501),
                             (epoch % 50 == 0, epoch % 50 == 0, epoch % 25 == 0))

    def test_nonfinal_epoch_does_not_trigger_extra_rollouts(self):
        self.assertEqual(self.events_at(798, 800), (False, False, False))
        self.assertEqual(self.events_at(775, 800), (False, False, True))
        self.assertEqual(self.events_at(749, 800), (True, True, False))

    def test_requested_final_three_evaluations_and_checkpoints(self):
        evaluations, checkpoints = [], []
        for epoch in range(801):
            rollout, checkpoint, _ = self.events_at(epoch, 801, final_n=30, final_interval=10)
            if rollout:
                evaluations.append(epoch)
            if checkpoint:
                checkpoints.append(epoch)
        expected = list(range(0, 751, 50)) + [780, 790, 800]
        self.assertEqual(evaluations, expected)
        self.assertEqual(checkpoints, expected)

    def test_dense_final_window_does_not_change_earlier_cadence(self):
        self.assertEqual(self.events_at(770, 801, final_n=30, final_interval=10)[:2], (False, False))
        self.assertEqual(self.events_at(779, 801, final_n=30, final_interval=10)[:2], (False, False))
        self.assertEqual(self.events_at(799, 801, final_n=30, final_interval=10)[:2], (False, False))

    def test_final_three_are_relative_to_arbitrary_total_epochs(self):
        for total in (30, 500, 501, 800, 801, 807, 1000):
            with self.subTest(total=total):
                evaluated = [epoch for epoch in range(total)
                             if self.events_at(epoch, total, final_n=30, final_interval=10)[0]]
                saved = [epoch for epoch in range(total)
                         if self.events_at(epoch, total, final_n=30, final_interval=10)[1]]
                expected = [total - 21, total - 11, total - 1]
                self.assertEqual(evaluated[-3:], expected)
                self.assertEqual(saved[-3:], expected)

    def test_short_debug_run_still_evaluates_and_saves_final_epoch(self):
        self.assertEqual(self.events_at(1, 2, final_n=30, final_interval=10), (True, True, True))


if __name__ == "__main__":
    unittest.main()
