# Adaptive Mean Flow

Core training and evaluation code for Adaptive Mean Flow, with transformer policy baselines.

Tasks: PushT (lowdim and image), and MetaWorld (lowdim).

## Install

```bash
conda env create -f env_specs/conda_environment_base.yaml
conda activate amf_base
pip install -e .
```

For MetaWorld, use `env_specs/conda_environment_metaworld.yaml` and activate `amf_metaworld` instead. These specs retain the original Python 3.10 / PyTorch 2.5.1 / CUDA 12.4 versions. MetaWorld uses the v3 environments; its dependency is pinned to the revision tested in the original development environment.

## Data

Run these commands from the repository root. Demonstrations and checkpoints are not bundled.

**PushT:** download the public Diffusion Policy dataset and extract it into `data/demos/`:

```bash
mkdir -p data/demos
curl -fL https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -o data/demos/pusht.zip
unzip data/demos/pusht.zip -d data/demos
```

**MetaWorld:** activate the MetaWorld environment and generate demonstrations with its scripted expert:

```bash
conda activate amf_metaworld
python scripts/generate_metaworld_demos.py --task pick-out-of-hole --num-demos 100
```

This saves `data/demos/metaworld/pick-out-of-hole/demo.zarr`, matching the training config. Replace the task name in both generation and training commands to use another task. Only successful episodes are saved; each attempt uses the next seed starting at zero and runs for at most 300 steps. Use `--seed`, `--max-steps`, or `--max-attempts` to override these settings. Existing output is never overwritten; choose `--output` for a different destination. Training uses the default path above.

## Train

From the repository root:

```bash
python train.py --config-name=train_adaptive_mean_flow_transformer_lowdim_workspace task=pusht/pusht_lowdim_abs
python train.py --config-name=train_adaptive_mean_flow_transformer_image_workspace task=pusht/pusht_image_abs
python train.py --config-name=train_adaptive_mean_flow_transformer_lowdim_workspace task=metaworld/metaworld_lowdim task.task=pick-out-of-hole
```

Choose `train_<policy>_transformer_<lowdim|image>_workspace`, where `<policy>` is `adaptive_mean_flow`, `flow_matching`, `adaptive_flow_matching`, `mean_flow`, or `streaming_diffusion`, using the supported combinations above. Use `pusht/pusht_lowdim_abs` or `pusht/pusht_image_abs` with the downloaded dataset, and `metaworld/metaworld_lowdim` with generated demonstrations.

Override settings on the command line, e.g. `training.seed=0 training.device=cuda:0`. Logs default to offline W&B; use `logging.mode=online` to enable upload. Checkpoints are saved under the run's `checkpoints/` directory. Configs provide starting settings, not a complete table of paper experiment hyperparameters.

## Evaluate

```bash
python eval.py --checkpoint /path/to/checkpoints/latest.ckpt --output_dir data/eval --device cuda:0
```

Evaluation uses the task, action horizon, and episode counts saved in the checkpoint and writes `eval_log.json` and configured rollout videos. Checkpoints must use the release's renamed Python class paths.

## Acknowledgments

The code is based on [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [Streaming Diffusion Policy](https://github.com/Streaming-Diffusion-Policy/streaming_diffusion_policy). Original copyright and license notices are retained in `LICENSE` and source files.
