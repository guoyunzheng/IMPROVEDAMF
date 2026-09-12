from typing import Optional
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import time
import wandb.sdk.data_types.video as wv
from source.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from source.gym_util.async_vector_env import AsyncVectorEnv
# from source.gym_util.sync_vector_env import SyncVectorEnv
from source.gym_util.multistep_wrapper import MultiStepWrapper
from source.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from source.policy.base_lowdim_policy import BaseLowdimPolicy
from source.common.pytorch_util import dict_apply
from source.env_runner.base_lowdim_runner import BaseLowdimRunner
from source.common.trajectory_smoothness import aggregate_smoothness_stats

def async_env_info_to_list(async_env_info, attr):
    """
        Utility function for extracting elements from returned info from AsyncVectorEnv.step().
        For some reason, this version of info() returns a vector with values from current and previous step
    """
    f = lambda x: x[attr][-1]
    return list(map(f,async_env_info))

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            abs_action=True,
            disturbance_force=0.0,
            save_trajectories=False,
            trajectory_dir='trajectories'
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        abs_action=abs_action,
                        disturbance_force=disturbance_force,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.save_trajectories = save_trajectories
        self.trajectory_dir = trajectory_dir

    def run(self, policy: BaseLowdimPolicy, high_level_policy: Optional[BaseLowdimPolicy] = None):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_prediction_times = []
        all_durations = [None] * n_inits
        all_action_trajectories = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()
            infos = None
            chunk_action_trajectories = [[] for _ in range(n_envs)]

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    start_predict = time.time()
                    action_dict = policy.predict_action(obs_dict)
                    end_predict = time.time()
                    all_prediction_times.append(end_predict - start_predict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                for env_idx in range(n_envs):
                    chunk_action_trajectories[env_idx].append(action[env_idx].copy())

                # step env
                obs, reward, done, infos = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_durations[this_global_slice] = async_env_info_to_list(infos,'time_step')[this_local_slice]
            for local_idx, global_idx in enumerate(range(start, end)):
                all_action_trajectories[global_idx] = np.concatenate(
                    chunk_action_trajectories[local_idx], axis=0)

        # import pdb; pdb.set_trace()

        if self.save_trajectories:
            trajectory_dir = pathlib.Path(self.output_dir).joinpath(self.trajectory_dir)
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            for i, trajectory in enumerate(all_action_trajectories):
                np.savez_compressed(
                    trajectory_dir.joinpath(
                        f"{self.env_prefixs[i].strip('/')}_seed_{self.env_seeds[i]}_actions.npz"),
                    action=trajectory,
                    seed=np.array(self.env_seeds[i], dtype=np.int64),
                    prefix=np.array(self.env_prefixs[i]))

        # log
        max_rewards = collections.defaultdict(list)
        durations = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            durations[prefix].append(all_durations[i])

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in durations.items():
            name = prefix+'mean_duration'
            value = np.mean(value)
            log_data[name] = value

        trajectories_by_prefix = collections.defaultdict(list)
        for i, trajectory in enumerate(all_action_trajectories):
            if trajectory is not None:
                trajectories_by_prefix[self.env_prefixs[i]].append(trajectory)
        for prefix, trajectories in trajectories_by_prefix.items():
            stats = aggregate_smoothness_stats(trajectories)
            for key, value in stats.items():
                log_data[prefix + 'action_' + key] = value

        # Log avg. prediction time
        log_data['mean_prediction_time'] = np.mean(all_prediction_times)

        return log_data

    def sweep_run(self, policy: BaseLowdimPolicy, sweep_param_dict: dict, reset_procedure=None):
        from itertools import product
        # Sweep rollouts over different inference parameters.
        # Using this instead of wandb sweep avoids retraining models only to test different inference params

        # Sweep param dict seperated into env_runner params and policy params
        combinations = list(product(
            *[[(cat, k, v) for v in values] for cat, params in sweep_param_dict.items() for k, values in params.items()]
            ))

        log_data = {}
        for comb in combinations:
            specifier = ""
            for seq in comb:
                specifier += seq[1] + str(seq[2]) + "_"
                if seq[0] == "env_runner":
                    def set_value(env):
                        setattr(env.unwrapped, seq[1], seq[2])
                    self.env.call("run_dill_function", dill.dumps(set_value),)
                    #setattr(self, seq[1], seq[2])
                elif seq[0] == "policy":
                    setattr(policy, seq[1], seq[2])

            curr_log_data = self.run(policy=policy)
            log_data[specifier + "/test/mean_score"] = curr_log_data["test/mean_score"]
            #log_data[specifier + "/train/mean_score"] = curr_log_data["train/mean_score"]
            log_data["mean_prediction_time"] = curr_log_data["mean_prediction_time"]
            if not isinstance(reset_procedure, type(None)):
                # If policy must be reset in some way between runs (e.g. reset obs buffers), this will be called here
                reset_procedure()

        best_test_score = -1.0
        #best_train_score = -1.0
        for key in log_data.keys():
            if key.find('test/mean_score') != -1:
                if log_data[key] > best_test_score:
                    best_test_score = log_data[key]
            # elif key.find('train/mean_score') != -1:
            #     if log_data[key] > best_train_score:
            #         best_train_score = log_data[key]

        log_data['test/mean_score'] = best_test_score
        #log_data['train/mean_score'] = best_train_score

        return log_data
