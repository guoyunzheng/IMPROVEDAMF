"""Run independent MetaWorld training jobs with bounded GPU concurrency.

Run from the repository root, in the amf environment:
    python scripts/train_metaworld_batch.py --dry-run
    python scripts/train_metaworld_batch.py
    python scripts/train_metaworld_batch.py --tasks reach push --jobs-per-gpu 1

Edit batch_configs/metaworld_amf.json to choose tasks, seeds and physical GPU
IDs. Missing demonstrations are generated sequentially before training. Existing
datasets must be complete; they are never overwritten. Each invocation creates
a new batch directory with manifest.json, summary.csv and one directory per
task/seed. Failures do not cancel unrelated jobs. Ctrl+C stops this launcher's
children, including their simulation workers. This is independent single-task
training, not a shared multi-task policy or distributed training.

Training defaults remain those in the README. The batch adds CPU thread limits,
explicit GPU assignment and independent output directories. It retains the top
three evaluated checkpoints by success rate, plus latest.ckpt. GPU capacity is
configured explicitly; the launcher does not reserve memory or detect other
users' workloads. MetaWorld still creates 50 simulation workers per training job.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "batch_configs/metaworld_amf.json"
CONFIG_KEYS = {
    "tasks", "seeds", "gpus", "jobs_per_gpu", "config_name", "num_epochs",
    "num_demos", "generate_missing_demos", "cpu_threads_per_job",
    "checkpoint_top_k", "output_root",
}
SUMMARY_FIELDS = [
    "task", "seed", "status", "gpu", "return_code", "last_epoch",
    "last_score", "best_score", "best_epoch", "last3_mean_score",
    "train_loss", "val_loss", "output_dir", "error",
]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def interrupt_batch(signum, frame):
    raise KeyboardInterrupt


def load_config(path, args=None):
    with Path(path).open(encoding="utf-8-sig") as stream:
        cfg = json.load(stream)
    if not isinstance(cfg, dict) or set(cfg) != CONFIG_KEYS:
        raise ValueError("Batch config must contain exactly: " + ", ".join(sorted(CONFIG_KEYS)))
    if args is not None:
        for key in ("tasks", "seeds", "gpus", "jobs_per_gpu"):
            value = getattr(args, key, None)
            if value is not None:
                cfg[key] = value
    for key in ("tasks", "seeds", "gpus"):
        values = cfg[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list")
    if any(not isinstance(task, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", task)
           or task.endswith("-v3") for task in cfg["tasks"]):
        raise ValueError("Use task names such as pick-place, without the -v3 suffix")
    if any(type(seed) is not int or seed < 0 for seed in cfg["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    if any(not isinstance(gpu, str) or not re.fullmatch(r"\d+", gpu) for gpu in cfg["gpus"]):
        raise ValueError('gpus must contain physical GPU indices as strings, e.g. ["0"]')
    cfg["gpus"] = [str(int(gpu)) for gpu in cfg["gpus"]]
    for key in ("tasks", "seeds", "gpus"):
        if len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"{key} contains duplicates")
    for key in ("jobs_per_gpu", "num_epochs", "num_demos", "cpu_threads_per_job", "checkpoint_top_k"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["generate_missing_demos"]) is not bool:
        raise ValueError("generate_missing_demos must be true or false")
    if not isinstance(cfg["config_name"], str) or not re.fullmatch(r"train_[a-z_]+_lowdim_workspace", cfg["config_name"]):
        raise ValueError("config_name must name a lowdim training config")
    if not (REPO_ROOT / "source/config" / (cfg["config_name"] + ".yaml")).is_file():
        raise ValueError(f"Unknown training config: {cfg['config_name']}")
    if not isinstance(cfg["output_root"], str) or not cfg["output_root"].strip():
        raise ValueError("output_root must be a nonempty path")
    return cfg


def child_environment(cfg, gpu=None):
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(cfg["cpu_threads_per_job"])
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "offline"
    env["HYDRA_FULL_ERROR"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def build_train_command(cfg, job):
    # JSON quoting protects spaces and Hydra grammar characters in the path.
    run_dir = json.dumps(Path(job["output_dir"]).as_posix(), ensure_ascii=False)
    return [
        sys.executable, "-u", str(REPO_ROOT / "train.py"),
        "--config-name=" + cfg["config_name"],
        "task=metaworld/metaworld_lowdim", "task.task=" + job["task"],
        f"training.seed={job['seed']}", "training.device=cuda:0",
        f"training.num_epochs={cfg['num_epochs']}", "training.resume=false",
        "logging.mode=offline", "checkpoint.topk.monitor_key=test_mean_score",
        "checkpoint.topk.mode=max", f"checkpoint.topk.k={cfg['checkpoint_top_k']}",
        "checkpoint.save_last_ckpt=true", "hydra.run.dir=" + run_dir,
    ]


def demo_command(cfg, task):
    return [sys.executable, "-u", str(REPO_ROOT / "scripts/generate_metaworld_demos.py"),
            "--task", task, "--num-demos", str(cfg["num_demos"])]


def validate_dataset(path, task, num_demos):
    import zarr
    root = zarr.open_group(str(path), mode="r")
    ends = root["meta/episode_ends"][:]
    obs = root["data/obs"]
    action = root["data/action"]
    valid = (
        root.attrs.get("complete") is True and root.attrs.get("task") == task
        and root.attrs.get("num_demos") == num_demos and len(ends) == num_demos
        and obs.shape == (int(ends[-1]), 21) and action.shape == (int(ends[-1]), 4)
        and int(ends[0]) > 0 and all(int(b) > int(a) for a, b in zip(ends, ends[1:]))
    )
    if not valid:
        raise ValueError(f"Incomplete or incompatible dataset: {path}. Expected {num_demos} successful {task} demos; existing data will not be overwritten.")


def read_metrics(path):
    metrics = {}
    scores = {}
    if not Path(path).is_file():
        return metrics
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            for source, target in (("epoch", "last_epoch"), ("train_loss", "train_loss"), ("val_loss", "val_loss")):
                value = row.get(source)
                if type(value) in (int, float) and math.isfinite(value):
                    metrics[target] = value
            score, epoch = row.get("test/mean_score"), row.get("epoch")
            if type(epoch) in (int, float) and math.isfinite(epoch) and type(score) in (int, float) and math.isfinite(score):
                scores[epoch] = score
    if scores:
        epochs = sorted(scores)
        best_epoch = max(epochs, key=lambda epoch: scores[epoch])
        last_three = [scores[epoch] for epoch in epochs[-3:]]
        metrics.update(last_score=scores[epochs[-1]], best_score=scores[best_epoch],
                       best_epoch=best_epoch, last3_mean_score=sum(last_three) / len(last_three))
    return metrics


def stop_processes(processes, grace_seconds=10):
    """Stop only process groups started by this launcher, never other jobs."""
    for proc in processes:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            elif proc.poll() is None:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while any(proc.poll() is None for proc in processes) and time.monotonic() < deadline:
        time.sleep(0.1)
    for proc in processes:
        try:
            if os.name == "posix":
                # The leader may exit before its simulation workers.
                os.killpg(proc.pid, signal.SIGKILL)
            elif proc.poll() is None:
                proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()


class BatchRunner:
    def __init__(self, cfg, batch_dir):
        self.cfg = cfg
        self.batch_dir = Path(batch_dir).resolve()
        self.jobs = [dict(task=task, seed=seed, status="pending", gpu=None,
                          return_code=None, output_dir=str(self.batch_dir / f"{task}_seed{seed}"), error="")
                     for task in cfg["tasks"] for seed in cfg["seeds"]]
        self.data = {task: {"status": "pending"} for task in cfg["tasks"]}
        self.state = dict(config=cfg, started_at=now(), status="preparing_data", datasets=self.data, jobs=self.jobs)
        self.active = []

    def persist(self):
        self.state["updated_at"] = now()
        temp = self.batch_dir / "manifest.json.tmp"
        temp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp.replace(self.batch_dir / "manifest.json")
        temp = self.batch_dir / "summary.csv.tmp"
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for job in self.jobs:
                writer.writerow({**job, **read_metrics(Path(job["output_dir"]) / "logs.json.txt")})
        temp.replace(self.batch_dir / "summary.csv")

    def start(self, record, command, log_path, gpu=None):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8")
        kwargs = {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        record.update(command=command, log_path=str(log_path))
        try:
            proc = subprocess.Popen(command, cwd=REPO_ROOT, env=child_environment(self.cfg, gpu),
                                    stdout=stream, stderr=subprocess.STDOUT, **kwargs)
        except BaseException:
            stream.close()
            raise
        entry = (proc, stream, record)
        self.active.append(entry)
        record.update(status="running", pid=proc.pid, started_at=now())
        self.persist()
        return entry

    def finish(self, entry):
        proc, stream, record = entry
        code = proc.wait()
        if code != 0:
            # A crashed training process can leave simulation workers behind.
            stop_processes([proc])
        stream.close()
        self.active.remove(entry)
        record.update(return_code=code, ended_at=now(), status="succeeded" if code == 0 else "failed")
        if code != 0:
            record["error"] = f"Process exited with code {code}; see {record['log_path']}"
        self.persist()
        return code

    def prepare_data(self):
        for task in self.cfg["tasks"]:
            record = self.data[task]
            path = REPO_ROOT / "data/demos/metaworld" / task / "demo.zarr"
            record["path"] = str(path)
            try:
                if not path.exists():
                    if not self.cfg["generate_missing_demos"]:
                        raise ValueError(f"Dataset missing: {path}")
                    print(f"[DATA] Generating {task}; log: {self.batch_dir / 'data_logs' / (task + '.log')}", flush=True)
                    entry = self.start(record, demo_command(self.cfg, task), self.batch_dir / "data_logs" / (task + ".log"))
                    if self.finish(entry) != 0:
                        raise RuntimeError(record["error"])
                validate_dataset(path, task, self.cfg["num_demos"])
                record["status"] = "ready"
                print(f"[DATA READY] {task}", flush=True)
            except Exception as exc:
                record.update(status="failed", error=str(exc))
                print(f"[DATA FAILED] {task}: {exc}", flush=True)
                for job in self.jobs:
                    if job["task"] == task:
                        job.update(status="blocked", error=str(exc))
            self.persist()

    def train(self):
        self.state["status"] = "training"
        pending = [job for job in self.jobs if job["status"] == "pending"]
        while pending or self.active:
            for gpu in self.cfg["gpus"]:
                count = sum(record.get("gpu") == gpu for _, _, record in self.active)
                while pending and count < self.cfg["jobs_per_gpu"]:
                    job = pending.pop(0)
                    job["gpu"] = gpu
                    try:
                        self.start(job, build_train_command(self.cfg, job), Path(job["output_dir"]) / "console.log", gpu)
                        count += 1
                        print(f"[START] {job['task']} seed={job['seed']} GPU={gpu}; {job['output_dir']}", flush=True)
                    except OSError as exc:
                        job.update(status="failed", error=str(exc))
                        self.persist()
            for entry in list(self.active):
                if entry[0].poll() is not None:
                    record = entry[2]
                    self.finish(entry)
                    print(f"[{record['status'].upper()}] {record['task']} seed={record['seed']}", flush=True)
            if pending or self.active:
                time.sleep(0.2)

    def run(self):
        # Fail rather than merge into an existing experiment directory.
        self.batch_dir.mkdir(parents=True, exist_ok=False)
        self.persist()
        exit_code = 1
        try:
            self.prepare_data()
            self.train()
            exit_code = 0 if all(job["status"] == "succeeded" for job in self.jobs) else 1
            self.state["status"] = "succeeded" if exit_code == 0 else "completed_with_errors"
        except KeyboardInterrupt:
            print("\nStopping this batch's child processes...", flush=True)
            self.state["status"] = "interrupted"
            exit_code = 130
        except Exception as exc:
            self.state.update(status="failed", error=str(exc))
            print(f"[BATCH FAILED] {exc}", file=sys.stderr, flush=True)
        finally:
            if self.active:
                stop_processes([entry[0] for entry in self.active])
                for proc, stream, record in self.active:
                    stream.close()
                    record.update(status="interrupted", return_code=proc.returncode, ended_at=now())
                self.active.clear()
            for record in [*self.jobs, *self.data.values()]:
                if record["status"] == "pending":
                    record["status"] = "cancelled"
            self.state["ended_at"] = now()
            self.persist()
        print(f"Batch status: {self.state['status']}\nResults: {self.batch_dir / 'summary.csv'}", flush=True)
        return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gpus", nargs="+", help="Physical GPU indices, e.g. --gpus 0 1")
    parser.add_argument("--jobs-per-gpu", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without importing ML packages, creating files or starting jobs")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, args)
        root = Path(cfg["output_root"])
        if not root.is_absolute():
            root = REPO_ROOT / root
        runner = BatchRunner(cfg, root / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        capacity = len(cfg["gpus"]) * cfg["jobs_per_gpu"]
        print(f"Tasks: {', '.join(cfg['tasks'])}; seeds: {cfg['seeds']}\nGPUs: {cfg['gpus']}; maximum concurrent training jobs: {capacity}\nEpochs per job: {cfg['num_epochs']}; simulation workers per job: 50\nBatch directory: {runner.batch_dir}", flush=True)
        if args.dry_run:
            for task in cfg["tasks"]:
                print("Reuse complete data, otherwise generate: " + shlex.join(demo_command(cfg, task)))
            for i, job in enumerate(runner.jobs):
                # Assignment during execution uses whichever GPU slot is free.
                gpu = cfg["gpus"][(i // cfg["jobs_per_gpu"]) % len(cfg["gpus"])]
                print(f"CUDA_VISIBLE_DEVICES={gpu} " + shlex.join(build_train_command(cfg, job)))
            return 0
        missing = [name for name in ("torch", "hydra", "zarr", "mujoco", "gymnasium", "metaworld")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError("Missing packages: " + ", ".join(missing) + ". Activate the prepared amf environment.")
        previous_handler = signal.signal(signal.SIGTERM, interrupt_batch)
        try:
            return runner.run()
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
