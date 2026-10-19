"""Compare recorded PushT keypoints with the current environment's definitions.

This is a CPU geometry check. It reads only state and keypoint arrays, does not
load model weights, and does not simulate dynamics or change the dataset.
"""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compare_keypoints(states, recorded, env_local):
    states = np.asarray(states, dtype=np.float64)
    recorded = np.asarray(recorded, dtype=np.float64)
    env_local = np.asarray(env_local, dtype=np.float64)
    if (states.ndim != 2 or states.shape[1] != 5 or len(states) == 0
            or recorded.shape != (len(states), len(env_local), 2)
            or env_local.ndim != 2 or env_local.shape[1] != 2
            or len(env_local) == 0):
        raise ValueError("Expected states (N,5), keypoints (N,K,2), local points (K,2)")
    if not all(np.isfinite(array).all() for array in (states, recorded, env_local)):
        raise ValueError("Geometry input contains NaN or Inf")

    # Undo the recorded block pose without resetting or stepping the simulator:
    # reset() applies legacy pose adjustments and a physics step, so it would
    # not recreate the exact measured pose for this coordinate comparison.
    delta = recorded - states[:, None, 2:4]
    cosine, sine = np.cos(states[:, 4, None]), np.sin(states[:, 4, None])
    local = np.stack((cosine * delta[..., 0] + sine * delta[..., 1],
                      -sine * delta[..., 0] + cosine * delta[..., 1]), axis=-1)
    recorded_local = np.median(local, axis=0)
    scatter = np.linalg.norm(local - recorded_local, axis=-1)
    ordered_errors = np.linalg.norm(local - env_local, axis=-1)
    cost = np.linalg.norm(recorded_local[:, None] - env_local[None, :], axis=-1)
    rows, columns = linear_sum_assignment(cost)
    mapping = np.empty(len(env_local), dtype=int)
    mapping[rows] = columns
    reordered_errors = np.linalg.norm(local - env_local[mapping], axis=-1)
    return {
        "frames_checked": len(states),
        "recorded_local_scatter_px_max": float(scatter.max()),
        "ordered_keypoint_error_px_mean": float(ordered_errors.mean()),
        "ordered_keypoint_error_px_max": float(ordered_errors.max()),
        "after_best_permutation_error_px_mean": float(reordered_errors.mean()),
        "after_best_permutation_error_px_max": float(reordered_errors.max()),
        "dataset_to_environment_keypoint_index": mapping.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/demos/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import zarr
    from source.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

    root = zarr.open_group(str(args.dataset.resolve()), mode="r")
    states = root["data/state"][:]
    keypoints = root["data/keypoint"][:]
    if len(states) != len(keypoints) or len(states) == 0:
        raise ValueError("State and keypoint arrays must have equal nonzero lengths")
    indices = np.linspace(0, len(states) - 1,
                          min(args.samples, len(states)), dtype=int)
    params = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    report = compare_keypoints(states[indices], keypoints[indices],
                               params["local_keypoint_map"]["block"])
    report["dataset"] = str(args.dataset.resolve())
    report["versions"] = {name: metadata.version(name)
                          for name in ("numpy", "scipy", "pygame", "pymunk", "scikit-image")}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
