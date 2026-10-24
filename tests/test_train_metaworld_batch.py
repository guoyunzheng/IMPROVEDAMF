"""Launcher tests without MuJoCo, CUDA, Hydra or training dependencies.

python -m unittest discover -s tests -p test_train_metaworld_batch.py -v
"""

import contextlib
import csv
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from scripts import train_metaworld_batch as batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        # Keep all temporary files within the repository's ignored data folder.
        parent = batch.REPO_ROOT / "data/test_metaworld_batch"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.parent == parent.resolve()
        self.addCleanup(self.temp.cleanup)
        self.cfg = batch.load_config(batch.DEFAULT_CONFIG)

    def config_file(self, **changes):
        path = self.root / "config.json"
        path.write_text(json.dumps({**self.cfg, **changes}), encoding="utf-8")
        return path

    def test_invalid_task_and_gpu_settings_rejected(self):
        for changes in ({"tasks": ["../reach"]}, {"tasks": ["reach-v3"]},
                        {"tasks": ["reach", "reach"]}, {"seeds": [True]},
                        {"gpus": ["0", "00"]}, {"gpus": ["0,1"]},
                        {"jobs_per_gpu": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                batch.load_config(self.config_file(**changes))

    def test_dry_run_does_not_create_output_or_spawn(self):
        output = self.root / "not-created"
        path = self.config_file(output_root=str(output))
        stream = io.StringIO()
        with patch.object(batch.subprocess, "Popen") as spawn, contextlib.redirect_stdout(stream):
            self.assertEqual(batch.main(["--config", str(path), "--dry-run"]), 0)
        self.assertFalse(output.exists())
        spawn.assert_not_called()
        self.assertIn("drawer-open", stream.getvalue())
        self.assertNotIn("pick-out-of-hole", stream.getvalue())

    def test_epoch_cli_overrides_reach_training_and_preserve_fresh_outputs(self):
        output = self.root / "epoch-control-dry-run"
        path = self.config_file(output_root=str(output), num_epochs=501, mf_epochs=100, flow_ratio=0.5)
        stream = io.StringIO()
        with patch.object(batch.subprocess, "Popen") as spawn, contextlib.redirect_stdout(stream):
            self.assertEqual(batch.main([
                "--config", str(path), "--tasks", "sweep", "--seeds", "43",
                "--num-epochs", "800", "--mf-epochs", "400", "--flow-ratio", "0.2", "--dry-run",
            ]), 0)
        spawn.assert_not_called()
        self.assertFalse(output.exists())
        text = stream.getvalue()
        for expected in ("pure FM=400, mixed FM/MF=400", "final epoch index=799",
                         "training.num_epochs=800", "training.resume=false",
                         "training.flow_ratio_schedule.enabled=true",
                         "++training.flow_ratio_schedule.fm_only_epochs=400", "flow_ratio=0.2"):
            self.assertIn(expected, text)

    def test_phase_json_settings_and_total_epoch_cli_override(self):
        path = self.config_file(num_epochs=501, mf_epochs=400, flow_ratio=0.2)
        cfg = batch.load_config(path, SimpleNamespace(num_epochs=801))
        command = batch.build_train_command(cfg, {"task": "sweep", "seed": 43, "output_dir": self.root})
        self.assertIn("training.num_epochs=801", command)
        self.assertIn("++training.flow_ratio_schedule.fm_only_epochs=401", command)

    def test_invalid_epoch_controls_rejected_before_launch(self):
        for changes in ({"num_epochs": 0}, {"mf_epochs": -1}, {"mf_epochs": 502},
                        {"mf_epochs": True}, {"mf_epochs": 400.5}, {"flow_ratio": True},
                        {"flow_ratio": -0.1}, {"flow_ratio": 1.1}, {"flow_ratio": float("nan")},
                        {"mf_epochs": 100, "config_name": "train_flow_matching_transformer_lowdim_workspace"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                batch.load_config(self.config_file(**changes))

    def test_existing_configs_keep_yaml_phase_settings(self):
        for path in (batch.REPO_ROOT / "batch_configs").glob("*.json"):
            with self.subTest(path=path):
                cfg = batch.load_config(path)
                command = batch.build_train_command(cfg, {"task": cfg["tasks"][0], "seed": 42, "output_dir": self.root})
                self.assertFalse(any("flow_ratio" in part for part in command))

    def test_separate_outputs_gpu_remapping_and_checkpoint_selection(self):
        cfg = {**self.cfg, "seeds": [0, 42]}
        runner = batch.BatchRunner(cfg, self.root / "batch with spaces")
        paths = [job["output_dir"] for job in runner.jobs]
        self.assertEqual(len(paths), len(set(paths)))
        command = batch.build_train_command(cfg, runner.jobs[0])
        self.assertIn("training.device=cuda:0", command)
        self.assertIn("checkpoint.topk.monitor_key=test_mean_score", command)
        self.assertIn("checkpoint.topk.k=3", command)
        self.assertIn("training.num_epochs=501", command)
        override = next(part for part in command if part.startswith("hydra.run.dir="))
        self.assertEqual(json.loads(override.split("=", 1)[1]), Path(paths[0]).as_posix())
        env = batch.child_environment(cfg, "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["WANDB_MODE"], "offline")

    def test_summary_uses_evaluation_rows_not_training_steps(self):
        path = self.root / "logs.json.txt"
        rows = [
            {"epoch": 0, "train_loss": 1.0, "test/mean_score": 0.0},
            {"epoch": 50, "test/mean_score": 0.9, "val_loss": 0.2},
            {"epoch": 100, "test/mean_score": 0.6},
            {"epoch": 100, "test/mean_score": 0.6},
            {"epoch": 150, "test/mean_score": 0.54, "train_loss": 0.07, "val_loss": 0.08},
            {"epoch": 151, "train_loss": 0.06, "test/mean_score": float("nan")},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + '\n{"truncated":', encoding="utf-8")
        result = batch.read_metrics(path)
        self.assertEqual(result["best_epoch"], 50)
        self.assertEqual(result["best_score"], 0.9)
        self.assertEqual(result["last_score"], 0.54)
        self.assertAlmostEqual(result["last3_mean_score"], (0.9 + 0.6 + 0.54) / 3)
        self.assertEqual(result["last_epoch"], 151)
        self.assertEqual(result["val_loss"], 0.08)

    def test_last3_uses_780_790_800_and_counts_each_epoch_once(self):
        path = self.root / "logs.json.txt"
        rows = [
            {"epoch": 700, "test/mean_score": 1.0},
            {"epoch": 750, "test/mean_score": 0.1},
            {"epoch": 780, "test/mean_score": 0.6},
            {"epoch": 790, "test/mean_score": 0.7},
            {"epoch": 790, "test/mean_score": 0.8},
            {"epoch": 799, "train_loss": 0.2},
            {"epoch": 800, "test/mean_score": 0.9},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        result = batch.read_metrics(path)
        self.assertEqual(result["last_epoch"], 800)
        self.assertEqual(result["last_score"], 0.9)
        self.assertAlmostEqual(result["last3_mean_score"], (0.6 + 0.8 + 0.9) / 3)
        self.assertEqual(result["best_score"], 1.0)
        self.assertEqual(result["best_epoch"], 700)

    def fake_dataset(self, complete=True, ends=(10, 20)):
        root = {
            "meta/episode_ends": list(ends),
            "data/obs": Mock(shape=(20, 21)),
            "data/action": Mock(shape=(20, 4)),
        }
        group = MagicMock()
        group.attrs = {"complete": complete, "task": "reach", "num_demos": 2}
        group.__getitem__ = Mock(side_effect=root.__getitem__)
        return group

    def test_dataset_completion_task_count_and_episode_boundaries(self):
        zarr = Mock()
        with patch.dict(sys.modules, {"zarr": zarr}):
            zarr.open_group.return_value = self.fake_dataset()
            batch.validate_dataset("unused", "reach", 2)
            for group, task, count in [
                (self.fake_dataset(complete=False), "reach", 2),
                (self.fake_dataset(), "push", 2),
                (self.fake_dataset(), "reach", 100),
                (self.fake_dataset(ends=(20, 10)), "reach", 2),
            ]:
                zarr.open_group.return_value = group
                with self.assertRaises(ValueError):
                    batch.validate_dataset("unused", task, count)

    def test_existing_incomplete_data_is_not_overwritten(self):
        cfg = {**self.cfg, "tasks": ["reach"]}
        runner = batch.BatchRunner(cfg, self.root / "batch")
        runner.batch_dir.mkdir()
        with patch.object(batch.Path, "exists", return_value=True), \
             patch.object(batch, "validate_dataset", side_effect=ValueError("incomplete")), \
             patch.object(runner, "start") as start, contextlib.redirect_stdout(io.StringIO()):
            runner.prepare_data()
        start.assert_not_called()
        self.assertEqual(runner.jobs[0]["status"], "blocked")
        self.assertEqual(runner.data["reach"]["status"], "failed")

    def test_dataset_failure_does_not_block_other_tasks(self):
        cfg = {**self.cfg, "tasks": ["reach", "push"]}
        runner = batch.BatchRunner(cfg, self.root / "batch")
        runner.batch_dir.mkdir()
        with patch.object(batch.Path, "exists", return_value=True), \
             patch.object(batch, "validate_dataset", side_effect=[ValueError("incomplete"), None]), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.prepare_data()
        self.assertEqual(runner.jobs[0]["status"], "blocked")
        self.assertEqual(runner.jobs[1]["status"], "pending")
        self.assertEqual(runner.data["push"]["status"], "ready")

    def test_actual_subprocess_queue_respects_slots_and_continues_after_failure(self):
        worker = self.root / "worker.py"
        worker.write_text('''import json, os, pathlib, sys, time
task, output, events = sys.argv[1:]
def event(kind):
    with open(events, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": kind, "task": task, "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "time": time.monotonic()}) + "\\n")
event("start")
print("worker", task, flush=True)
time.sleep(1.0)
pathlib.Path(output, "logs.json.txt").write_text(json.dumps({"epoch": 500, "test/mean_score": 0.54}) + "\\n")
event("end")
sys.exit(7 if task == "push" else 0)
''', encoding="utf-8")
        eight_slot_tasks = self.cfg["tasks"] + [
            "door-open", "button-press", "window-open", "drawer-close",
            "door-close", "window-close", "faucet-open", "faucet-close",
        ]
        for gpus, slots, tasks in [(["0", "1"], 1, self.cfg["tasks"]),
                                   (["0"], 2, self.cfg["tasks"]),
                                   (["0"], 8, eight_slot_tasks)]:
            with self.subTest(gpus=gpus, slots=slots):
                cfg = {**self.cfg, "gpus": gpus, "jobs_per_gpu": slots, "tasks": tasks}
                runner = batch.BatchRunner(cfg, self.root / ("batch_" + "_".join(gpus) + "_" + str(slots)))
                def prepared():
                    for record in runner.data.values():
                        record["status"] = "ready"
                def command(config, job):
                    # Each process owns its log; shared append writes can race on Windows.
                    events_path = Path(job["output_dir"]) / "events.jsonl"
                    return [sys.executable, "-u", str(worker), job["task"], job["output_dir"], str(events_path)]
                with patch.object(runner, "prepare_data", side_effect=prepared), \
                     patch.object(batch, "build_train_command", side_effect=command), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.run(), 1)
                self.assertEqual([job["status"] for job in runner.jobs],
                                 ["failed" if task == "push" else "succeeded" for task in tasks])
                events = sorted((json.loads(line)
                                 for job in runner.jobs
                                 for line in (Path(job["output_dir"]) / "events.jsonl")
                                 .read_text(encoding="utf-8").splitlines()), key=lambda e: e["time"])
                active = {gpu: 0 for gpu in gpus}
                peak = 0
                for event in events:
                    active[event["gpu"]] += 1 if event["kind"] == "start" else -1
                    self.assertLessEqual(active[event["gpu"]], slots)
                    self.assertGreaterEqual(active[event["gpu"]], 0)
                    peak = max(peak, sum(active.values()))
                self.assertEqual(peak, min(len(tasks), slots * len(gpus)))
                self.assertEqual(sum(active.values()), 0)
                for job in runner.jobs:
                    self.assertTrue((Path(job["output_dir"]) / "console.log").is_file())
                with (runner.batch_dir / "summary.csv").open(encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), len(tasks))
                self.assertEqual(rows[1]["return_code"], "7")
                self.assertEqual(rows[0]["last_score"], "0.54")
                state = json.loads((runner.batch_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "completed_with_errors")

    def test_interruption_marks_pending_jobs_and_cleans_owned_processes(self):
        runner = batch.BatchRunner(self.cfg, self.root / "interrupted")
        proc = Mock(pid=12345, returncode=-15)
        stream = Mock()
        def interrupt():
            runner.jobs[0]["status"] = "running"
            runner.active.append((proc, stream, runner.jobs[0]))
            raise KeyboardInterrupt
        with patch.object(runner, "prepare_data", side_effect=interrupt), \
             patch.object(batch, "stop_processes") as stop, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(), 130)
        stop.assert_called_once_with([proc])
        stream.close.assert_called_once()
        self.assertEqual(runner.jobs[0]["status"], "interrupted")
        self.assertTrue(all(job["status"] == "cancelled" for job in runner.jobs[1:]))

    def test_posix_cleanup_signals_only_owned_groups(self):
        processes = [Mock(pid=12345), Mock(pid=12346)]
        with patch.object(batch.os, "name", "posix"), \
             patch.object(batch.os, "killpg", create=True) as kill, \
             patch.object(batch.signal, "SIGKILL", 9, create=True):
            batch.stop_processes(processes, grace_seconds=0)
        self.assertEqual(kill.call_count, 4)
        self.assertEqual({call.args[0] for call in kill.call_args_list}, {12345, 12346})
        self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGTERM, signal.SIGTERM, 9, 9])

    def test_existing_batch_directory_is_never_reused(self):
        runner = batch.BatchRunner(self.cfg, self.root)
        with self.assertRaises(FileExistsError):
            runner.run()


if __name__ == "__main__":
    unittest.main()
