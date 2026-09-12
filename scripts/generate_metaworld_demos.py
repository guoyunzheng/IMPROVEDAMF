"""Generate successful MetaWorld v3 scripted-policy demonstrations in Zarr format."""

import argparse
import importlib
from pathlib import Path

import gymnasium as gym
import metaworld  # noqa: F401 — registers Meta-World environments
import numpy as np
import zarr


def normalize_task(task):
    """Accept reach, reach-v3, and reach_v3; return the canonical task name."""
    task = task.strip().lower().replace("_", "-")
    if task.endswith("-v3"):
        task = task[:-3]
    if not task or any(not part.isalnum() for part in task.split("-")):
        raise ValueError(f"Invalid task name: {task!r}")
    return task


def get_policy(task):
    base = normalize_task(task).replace("-", "_")
    module = importlib.import_module(f"metaworld.policies.sawyer_{base}_v3_policy")
    class_name = "Sawyer" + "".join(part.capitalize() for part in base.split("_")) + "V3Policy"
    return getattr(module, class_name)()


def generate_demos(task, num_demos=100, output=None, max_steps=300,
                   seed=0, max_attempts=None, policy_name=None):
    """Record (obs_t, action_t), keeping only episodes that reach success.

    Seeds advance on every attempt, including failures. Observations contain
    the first 18 current-state coordinates and the final three goal coordinates,
    matching MetaworldLowdimWrapper. Partial output is retained if attempts run out.
    """
    task = normalize_task(task)
    if num_demos <= 0 or max_steps <= 0 or seed < 0:
        raise ValueError("num_demos and max_steps must be positive; seed must be nonnegative")
    max_attempts = num_demos * 100 if max_attempts is None else max_attempts
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    output = Path(output) if output is not None else Path("data/demos/metaworld") / task / "demo.zarr"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output path.")

    policy = get_policy(policy_name or task)
    env = gym.make("Meta-World/MT1", env_name=f"{task}-v3")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(output), mode="w-")
        data = root.create_group("data")
        obs_arr = data.create_dataset("obs", shape=(0, 21), chunks=(2048, 21), dtype="float32")
        act_arr = data.create_dataset("action", shape=(0, 4), chunks=(2048, 4), dtype="float32")
        ends = root.create_group("meta").create_dataset(
            "episode_ends", shape=(0,), chunks=(1024,), dtype="int64")
        root.attrs.update(task=task, env_name=f"{task}-v3", num_demos=0,
                          requested_num_demos=num_demos, max_steps=max_steps,
                          start_seed=seed, successful_seeds=[], complete=False)
        successful_seeds = []
        for attempt in range(max_attempts):
            episode_seed = seed + attempt
            env.unwrapped.seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            observations, actions = [], []
            success = False
            for _ in range(max_steps):
                action = np.asarray(policy.get_action(obs), dtype=np.float32)
                lowdim_obs = np.concatenate((obs[:18], obs[-3:])).astype(np.float32)
                if lowdim_obs.shape != (21,) or action.shape != (4,):
                    raise ValueError("Expected MetaWorld v3 observations and four-dimensional actions")
                observations.append(lowdim_obs)
                actions.append(action)
                obs, _, terminated, truncated, info = env.step(action)
                success = bool(info.get("success", False))
                if success or terminated or truncated:
                    break
            if success:
                obs_arr.append(np.stack(observations), axis=0)
                act_arr.append(np.stack(actions), axis=0)
                ends.append(np.array([obs_arr.shape[0]], dtype=np.int64))
                successful_seeds.append(episode_seed)
                root.attrs.update(num_demos=len(successful_seeds), successful_seeds=successful_seeds)
            root.attrs["attempts"] = attempt + 1
            print(f"Seed {episode_seed}: {'success' if success else 'failed'} "
                  f"({len(successful_seeds)}/{num_demos} demos)", flush=True)
            if len(successful_seeds) == num_demos:
                root.attrs["complete"] = True
                return output
        raise RuntimeError(f"Collected {len(successful_seeds)}/{num_demos} successful demos "
                           f"in {max_attempts} attempts. Partial dataset: {output}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="MetaWorld task, e.g. pick-out-of-hole")
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0, help="First attempted episode seed")
    parser.add_argument("--max-attempts", type=int, help="Defaults to 100 times --num-demos")
    parser.add_argument("--output", type=Path, help="Defaults to data/demos/metaworld/<task>/demo.zarr")
    parser.add_argument("--policy-name", help="Optional scripted-policy task name override")
    args = parser.parse_args()
    output = generate_demos(**vars(args))
    print(f"Wrote demos to {output}")


if __name__ == "__main__":
    main()
