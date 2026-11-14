"""Fixed-midpoint Kinematic MeanFlow targets (Fan et al., Eq. 7).

The policy call convention here is model(z, t, r, cond). Times are either
trajectory scalars [B] for MF or per-action vectors [B, H] for AMF.
"""

import torch


def midpoint_target(model, actions, noise, t, r, cond):
    """Build a detached target using two independently evaluated subintervals.

    Both states lie on the conditional path through the same actions and noise.
    Each JVP varies the state and upper time, holding the lower time fixed.
    The 1/4 weights combine the interval weight and its relative duration.
    """
    with torch.no_grad():
        midpoint = (t + r) * 0.5
        time_shape = (t.shape[0], -1, 1)
        t_broadcast = t.reshape(time_shape)
        midpoint_broadcast = midpoint.reshape(time_shape)
        velocity = noise - actions
        z_t = (1 - t_broadcast) * actions + t_broadcast * noise
        z_midpoint = (1 - midpoint_broadcast) * actions + midpoint_broadcast * noise

        states = torch.cat((z_t, z_midpoint), dim=0)
        upper_times = torch.cat((t, midpoint), dim=0)
        lower_times = torch.cat((midpoint, r), dim=0)
        velocities = torch.cat((velocity, velocity), dim=0)
        conditions = torch.cat((cond, cond), dim=0)

        # Reuse the repository's JVP backend. The target needs no backward graph.
        _, derivatives = torch.autograd.functional.jvp(
            lambda state, upper, lower: model(state, upper, lower, cond=conditions),
            (states, upper_times, lower_times),
            (velocities, torch.ones_like(upper_times), torch.zeros_like(lower_times)),
            create_graph=False,
        )
        # AMP can return half-precision JVPs even when actions are float32.
        # Promote before summing, so two finite derivatives cannot overflow here.
        target_dtype = torch.promote_types(velocity.dtype, torch.float32)
        first_derivative, second_derivative = derivatives.to(target_dtype).chunk(2, dim=0)
        gap = (t - r).reshape(time_shape)
        target = velocity - 0.25 * gap * (first_derivative + second_derivative)
    return target
