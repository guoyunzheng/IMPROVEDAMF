"""CPU smoke checks: run `python tests/smoke_policies.py` from the repository root.

Resolve every supported config combination and test loss/backpropagation plus
repeated inference for every policy, using small transformer models.
"""
from pathlib import Path
import sys, importlib, ast
r = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(r))
import hydra, torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
OmegaConf.register_new_resolver('eval', eval, replace=True)
torch.set_num_threads(2)
configs = sorted((r / 'source/config').glob('train_*.yaml'))
with initialize_config_dir(config_dir=str(r / 'source/config'), version_base=None):
    for p in configs:
        image = '_image_' in p.name
        tasks = ['pusht/pusht_image', 'pusht/pusht_image_abs'] if image else ['pusht/pusht_lowdim', 'pusht/pusht_lowdim_abs', 'metaworld/metaworld_lowdim']
        for task in tasks:
            cfg = compose(config_name=p.stem, overrides=['task=' + task])
            OmegaConf.resolve(cfg)
            hydra.utils.get_class(cfg._target_)
            hydra.utils.get_class(cfg.task.dataset._target_)
            if 'metaworld' not in task:
                hydra.utils.get_class(cfg.task.env_runner._target_)
        prefix = 'policy.model.' if p.stem == 'train_flow_matching_transformer_lowdim_workspace' else 'policy.'
        cfg = compose(config_name=p.stem, overrides=['task=' + tasks[0], prefix + 'n_layer=1', prefix + 'n_emb=32', prefix + 'n_head=4'])
        OmegaConf.resolve(cfg)
        policy = hydra.utils.instantiate(cfg.policy)
        from source.model.common.normalizer import LinearNormalizer
        B = 2
        T = cfg.horizon
        To = cfg.n_obs_steps
        if image:
            obs = {k: torch.rand(B, T, *v.shape) for (k, v) in cfg.task.shape_meta.obs.items()}
            data = {'action': torch.rand(B, T, 2), 'agent_pos': obs['agent_pos']}
            norm = LinearNormalizer()
            norm.fit(data, last_n_dims=1)
            from source.common.normalize_util import get_image_range_normalizer
            norm['image'] = get_image_range_normalizer()
            batch = {'obs': obs, 'action': data['action']}
        else:
            batch = {'obs': torch.rand(B, T, cfg.obs_dim), 'action': torch.rand(B, T, cfg.action_dim)}
            norm = LinearNormalizer()
            norm.fit(batch, last_n_dims=1)
        policy.set_normalizer(norm)
        loss = policy.compute_loss(batch)
        loss = loss[0] if isinstance(loss, tuple) else loss
        assert torch.isfinite(loss)
        loss.backward()
        policy.eval()
        input_obs = {k: v[:, :To] for (k, v) in batch['obs'].items()} if image else {'obs': batch['obs'][:, :To]}
        with torch.no_grad():
            for _ in range(2):
                out = policy.predict_action(input_obs)
                assert torch.isfinite(out['action']).all()
        print('PASS', p.stem, flush=True)
print('All 21 config combinations and 8 policy forward/backward/inference checks passed.')
