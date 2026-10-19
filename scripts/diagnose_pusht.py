"""Inspect a PushT run and optionally compare its raw and EMA checkpoints.

Run from the repository root. Evaluation preserves the checkpoint's action
representation, noise schedule, episode seeds and episode length. No training
or checkpoint changes are performed. Results go to a new diagnostics directory.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_evaluations(path):
    evaluations = []
    last = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
                if "test/mean_score" in last:
                    evaluations.append({key: last[key] for key in (
                        "epoch", "train_loss", "val_loss", "train/mean_score",
                        "test/mean_score", "train_action_mse_error",
                    ) if key in last})
    return last, evaluations


def classify_nonfinite_state(state):
    """Separate valid additive attention masks from unexpected NaN/Inf values."""
    import torch

    unexpected = []
    attention_masks = []
    for key, value in state.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        if torch.isfinite(value).all().item():
            continue
        # These registered buffers use 0 for visible positions and -inf for
        # blocked positions. Do not exempt weights, NaN, +inf, or fully blocked
        # rows: those can still make attention produce invalid outputs.
        valid_mask = (
            key in {"model.mask", "model.memory_mask"}
            and value.ndim == 2 and value.numel() > 0
            and ((value == 0) | torch.isneginf(value)).all().item()
            and (value == 0).any(dim=-1).all().item()
        )
        (attention_masks if valid_mask else unexpected).append(key)
    return unexpected, attention_masks


def inspect_run(run, payload, cfg):
    import numpy as np
    import torch
    import hydra
    from omegaconf import OmegaConf

    fields = (
        "task.name", "task.dataset.abs_action", "task.env_runner.abs_action",
        "task.dataset.max_train_episodes", "task.dataset.seed",
        "training.num_epochs", "training.debug", "training.seed",
        "training.max_train_steps", "training.max_val_steps", "training.use_ema",
        "training.rollout_every", "training.checkpoint_every",
        "training.last_n_rollout_every.interval", "dataloader.batch_size",
        "horizon", "n_obs_steps", "n_action_steps", "policy.num_inference_steps",
        "policy.flow_matching_inference", "policy.flow_ratio",
        "policy.monotone_schedule_ratio", "policy.t_start", "policy.t_end",
        "policy.p_curve", "task.env_runner.n_train", "task.env_runner.n_test",
        "task.env_runner.n_envs", "task.env_runner.test_start_seed",
        "task.env_runner.max_steps", "task.env_runner.legacy_test",
        "task.env_runner.disturbance_force", "checkpoint.topk.monitor_key",
    )
    last, evaluations = read_evaluations(run / "logs.json.txt")
    report = {
        "run_dir": str(run),
        "torch_version": torch.__version__,
        "config": {key: OmegaConf.select(cfg, key) for key in fields},
        "last_epoch": last.get("epoch") if last else None,
        "evaluations": evaluations,
        "normalizer_check": {},
        "nonfinite_state_keys": {},
        "expected_attention_mask_keys": {},
    }
    if evaluations:
        report["best_logged_evaluation"] = max(
            evaluations, key=lambda row: row["test/mean_score"])

    # Inspect the actual dataset representation used to train the checkpoint.
    # Lowdim PushT only loads keypoints, states and actions, not image arrays.
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    actions = dataset._sample_to_data(dataset.replay_buffer)["action"]
    actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    report["dataset"] = {
        "episodes": int(dataset.replay_buffer.n_episodes),
        "training_episodes": int(dataset.train_mask.sum()),
        "training_windows": len(dataset),
        "action_mean": actions.mean(dim=0).tolist(),
        "action_std": actions.std(dim=0).tolist(),
    }
    for name in ("model", "ema_model"):
        state = payload["state_dicts"].get(name)
        if state is None:
            continue
        unexpected, masks = classify_nonfinite_state(state)
        report["nonfinite_state_keys"][name] = unexpected
        report["expected_attention_mask_keys"][name] = masks
        prefix = "normalizer.params_dict.action."
        scale, offset = state[prefix + "scale"], state[prefix + "offset"]
        normalized = actions * scale + offset
        report["normalizer_check"][name] = {
            "saved_input_mean": state[prefix + "input_stats.mean"].tolist(),
            "saved_input_std": state[prefix + "input_stats.std"].tolist(),
            "normalized_action_mean": normalized.mean(dim=0).tolist(),
            "normalized_action_std": normalized.std(dim=0).tolist(),
        }
    return report


def evaluate(payload, cfg, output, device, seed, videos, weights):
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the inspection without --evaluate first.")
    # Match the attention implementation selected by the training workspace.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    names = {"both": ("ema_model", "model"), "ema": ("ema_model",),
             "raw": ("model",)}[weights]
    for name in names:
        if name not in payload["state_dicts"]:
            raise ValueError(f"Checkpoint has no {name} weights")

    results = {}
    for name in names:
        subdir = output / name
        subdir.mkdir()
        policy = hydra.utils.instantiate(cfg.policy)
        policy.load_state_dict(payload["state_dicts"][name], strict=True)
        policy.to(device).eval()
        policy.reset()
        if hasattr(policy, "reset_history"):
            policy.reset_history()

        runner_cfg = OmegaConf.create(OmegaConf.to_container(
            cfg.task.env_runner, resolve=True))
        runner_cfg.n_train_vis = 0
        runner_cfg.n_test_vis = min(videos, runner_cfg.n_test)
        # One batch avoids carrying cached trajectories between episode batches.
        runner_cfg.n_envs = runner_cfg.n_train + runner_cfg.n_test
        OmegaConf.save(runner_cfg, subdir / "runner_config.yaml")
        runner = hydra.utils.instantiate(runner_cfg, output_dir=str(subdir))
        try:
            # Reset AFTER model/environment construction so both weights receive
            # the same random stream. This differs from training's RNG state.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                metrics = runner.run(policy)
            serializable = {
                key: str(value._path) if hasattr(value, "_path") else value
                for key, value in metrics.items()
            }
            (subdir / "eval_log.json").write_text(
                json.dumps(serializable, indent=2), encoding="utf-8")
            results[name] = {
                "test/mean_score": float(metrics["test/mean_score"]),
                "train/mean_score": metrics.get("train/mean_score"),
                "video_paths": {key: value for key, value in serializable.items()
                                if "sim_video" in key},
            }
            (output / "comparison.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
            print(f"{name}: test/mean_score={metrics['test/mean_score']:.4f}", flush=True)
        finally:
            runner.env.close()
            del policy, runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="Policy sampling seed, not episode seeds")
    parser.add_argument("--videos", type=int, default=2, help="Test videos per weight variant")
    parser.add_argument("--weights", choices=("both", "ema", "raw"), default="both")
    args = parser.parse_args()
    if args.videos < 0 or args.seed < 0:
        parser.error("--videos and --seed must be nonnegative")
    run = args.run_dir.resolve()
    os.chdir(ROOT)
    for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                       "MPLBACKEND": "Agg", "WANDB_MODE": "offline"}.items():
        os.environ.setdefault(key, value)

    import dill
    import torch
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(run / "checkpoints/latest.ckpt", map_location="cpu",
                         pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    if cfg.task.env_runner._target_ != "source.env_runner.pusht_keypoints_runner.PushTKeypointsRunner":
        raise ValueError("This diagnostic supports lowdim PushT checkpoints only")
    report = inspect_run(run, payload, cfg)
    print(json.dumps(report, indent=2), flush=True)
    if not args.evaluate:
        return
    if any(report["nonfinite_state_keys"].values()):
        raise ValueError("Nonfinite checkpoint tensors detected; inspect the report before evaluation")
    if cfg.task.dataset.abs_action != cfg.task.env_runner.abs_action:
        raise ValueError("Dataset and runner action representations disagree")
    output = run / "diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report["evaluation_sampling_seed"] = args.seed
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Diagnostic output: {output}", flush=True)
    results = evaluate(payload, cfg, output, args.device, args.seed, args.videos, args.weights)
    print(json.dumps(results, indent=2), flush=True)
    print(f"Diagnostic output: {output}", flush=True)


if __name__ == "__main__":
    main()
