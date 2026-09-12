import numpy as np


def finite_difference_norms(trajectory, order):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2:
        raise ValueError(f"Expected trajectory with shape (T, D), got {trajectory.shape}")
    if trajectory.shape[0] <= order:
        return np.empty((0,), dtype=np.float64)
    diff = np.diff(trajectory, n=order, axis=0)
    return np.linalg.norm(diff, axis=-1)


def trajectory_smoothness_stats(trajectory):
    accel = finite_difference_norms(trajectory, order=2)
    jerk = finite_difference_norms(trajectory, order=3)
    return {
        "acceleration_mean": float(np.mean(accel)) if accel.size else np.nan,
        "acceleration_median": float(np.median(accel)) if accel.size else np.nan,
        "acceleration_max": float(np.max(accel)) if accel.size else np.nan,
        "jerk_mean": float(np.mean(jerk)) if jerk.size else np.nan,
        "jerk_median": float(np.median(jerk)) if jerk.size else np.nan,
        "jerk_max": float(np.max(jerk)) if jerk.size else np.nan,
    }


def aggregate_smoothness_stats(trajectories):
    all_accel = list()
    all_jerk = list()
    for trajectory in trajectories:
        all_accel.append(finite_difference_norms(trajectory, order=2))
        all_jerk.append(finite_difference_norms(trajectory, order=3))

    accel = np.concatenate([x for x in all_accel if x.size]) if any(x.size for x in all_accel) else np.empty((0,))
    jerk = np.concatenate([x for x in all_jerk if x.size]) if any(x.size for x in all_jerk) else np.empty((0,))

    return {
        "acceleration_mean": float(np.mean(accel)) if accel.size else np.nan,
        "acceleration_median": float(np.median(accel)) if accel.size else np.nan,
        "acceleration_max": float(np.max(accel)) if accel.size else np.nan,
        "jerk_mean": float(np.mean(jerk)) if jerk.size else np.nan,
        "jerk_median": float(np.median(jerk)) if jerk.size else np.nan,
        "jerk_max": float(np.max(jerk)) if jerk.size else np.nan,
    }
