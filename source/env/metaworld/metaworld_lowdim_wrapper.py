import gym
import gymnasium as gymn
import metaworld
import gym.spaces as gym_spaces
import gymnasium.spaces as gymn_spaces
from gymnasium.utils import seeding
import numpy as np
import random

def gymnasium_space_to_gym(space: gymn_spaces.Space) -> gym_spaces.Space:
    """
    Recursively convert a gymnasium.Space into an equivalent gym.Space.

    Supported: Dict, Box, Discrete, MultiDiscrete, MultiBinary, Tuple.

    Notes:
      - Box.low/high are copied to NumPy arrays to avoid shared references.
      - Discrete: gymnasium.Discrete has optional (start) in newer versions; gym.Discrete
        historically assumes start=0. If a non-zero start is present, this function errors
        to avoid silently changing semantics.
    """
    if isinstance(space, gymn_spaces.Dict):
        return gym_spaces.Dict({
            key: gymnasium_space_to_gym(subspace)
            for key, subspace in space.spaces.items()
        })

    if isinstance(space, gymn_spaces.Box):
        return gym_spaces.Box(
            low=np.asarray(space.low),
            high=np.asarray(space.high),
            #shape=space.shape,
            dtype=space.dtype,
        )

    if isinstance(space, gymn_spaces.Discrete):
        # Gymnasium Discrete may expose a `start` attribute; Gym's Discrete is (typically) start=0.
        start = getattr(space, "start", 0)
        if start != 0:
            raise ValueError(
                f"Cannot safely convert gymnasium.spaces.Discrete(start={start}) to gym.spaces.Discrete "
                "without changing semantics. Consider wrapping/remapping or ensuring start=0."
            )
        return gym_spaces.Discrete(space.n)

    if isinstance(space, gymn_spaces.MultiDiscrete):
        return gym_spaces.MultiDiscrete(space.nvec)

    if isinstance(space, gymn_spaces.MultiBinary):
        return gym_spaces.MultiBinary(space.n)

    if isinstance(space, gymn_spaces.Tuple):
        return gym_spaces.Tuple(tuple(gymnasium_space_to_gym(s) for s in space.spaces))

    raise TypeError(f"Unsupported gymnasium space type: {type(space).__name__}")


class MetaworldLowdimWrapper(gym.Env):
    def __init__(self,
                 task,
                 action_noise_magnitude=0.0,
                 ac_metrics=None):

        render_mode = 'rgb_array'
        env_name = f"{task}-v3"
        env = gymn.make('Meta-World/MT1', env_name=env_name, render_mode=render_mode, width=96, height=96)
        self.env = env
        self.anm = action_noise_magnitude
        self.an_rho = 0.60


        gymn_obs_space = env.observation_space
        obs_low = np.concatenate((gymn_obs_space.low[:18], gymn_obs_space.low[-3:]))
        obs_high = np.concatenate((gymn_obs_space.high[:18], gymn_obs_space.high[-3:]))
        self.observation_space = gym_spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=gymn_obs_space.dtype
        )
        self.action_space = gymnasium_space_to_gym(env.action_space)
        self.ac_noise = np.zeros(self.action_space.shape, dtype=np.float32)

        if self.anm > 0.0:
            assert ac_metrics is not None, "Action noise magnitude > 0.0 requires ac_metrics with action mean/std for scaling."
            self.act_mean = ac_metrics['mean']
            self.act_std = ac_metrics['std']
        else:
            self.act_mean = np.zeros(self.action_space.shape, dtype=np.float32)
            self.act_std = np.ones(self.action_space.shape, dtype=np.float32)

        self._seed = None

    def seed(self, seed=None):
        if seed is None:
            seed = random.randint(0,9999999)
        self._seed = seed
        self.env.unwrapped.seed(seed)

    def step(self, action):

        # Add ou noise to action
        if self.anm > 0.0:
            self.ac_noise = self.an_rho * self.ac_noise + np.sqrt(1 - self.an_rho**2) * np.random.normal(size=self.action_space.shape, scale=self.act_std, loc=self.act_mean) * self.anm
            action = action + self.ac_noise

        raw_obs, _, _, _, info = self.env.step(action)
        obs = np.concatenate((raw_obs[:18], raw_obs[-3:]))
        reward = float(info['success'])
        done = reward == 1.0
        self.time_step += 1
        info['time_step'] = self.time_step
        return obs, reward, done, info

    def render(self, mode=None):
        return self.env.render()

    def reset(self):
        raw_obs,_ = self.env.reset()
        obs = np.concatenate((raw_obs[:18], raw_obs[-3:]))
        self.time_step = 0
        self.ac_noise[:] = 0.0
        return obs
