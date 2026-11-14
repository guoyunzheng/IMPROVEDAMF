# K-MF 固定中点训练目标

本实现依据 Kinematic MeanFlow 论文式 7 和算法 1，提供固定中点版本。
2026-11-14 检查官方仓库时，README 标注代码正在准备发布，目录包含 README 和图片。
本实现由论文公式独立实现并适配本项目。

- [论文](https://arxiv.org/abs/2610.00864)
- [官方仓库](https://github.com/IntelChina-AI/K-MF)

四类 MF/AMF 的 lowdim、image 策略均支持 `policy.mf_target_type`。
默认值 `meanflow` 使用原有目标，`kmf_midpoint` 启用固定中点目标。
旧配置省略该字段时使用 `meanflow`。

给定归一化动作 `a`、同形状噪声 `e` 和采样时间 `t >= r`，令

```text
c   = (t + r) / 2
z_t = (1 - t) * a + t * e
z_c = (1 - c) * a + c * e
v   = e - a

D_tc = JVP(model, (z_t, t, c), (v, 1, 0))
D_cr = JVP(model, (z_c, c, r), (v, 1, 0))
target = stop_gradient(v - (t - r) * (D_tc + D_cr) / 4)
error  = model(z_t, t, r, cond) - target
```

这里按本项目接口顺序书写 `model(z, t, r, cond)`。两段导数使用同一个模型，
条件特征沿 batch 维复制，两段 JVP 合并计算。`z_c` 直接由训练动作和噪声构造。
目标计算使用 `torch.autograd.functional.jvp(create_graph=False)`，
两段导数求和时至少使用 float32。主预测保留反向传播，包括图像编码器的梯度。

MF 使用每条轨迹一个时间值 `[B]`。AMF 对 `[B,H]` 中每个动作位置分别取中点，
沿用项目已有的方向 JVP。这是对 AMF 向量时间的适配，论文推导采用标量时间。

回归目标使用项目原有的自适应 L2 归一化。论文式 7 写作平方误差，
因此本实现的外层损失采用本项目约定。AMF lowdim 的 progressive gap 权重
施加在归一化后的逐轨迹误差上。纯 FM 预热时 `r=t`，直接回归 `v` 并跳过 JVP。
采样分布与 FM/MF 阶段继续由已有训练配置控制。

单任务训练可在仓库根目录运行

```bash
python train.py --config-name=train_adaptive_mean_flow_transformer_lowdim_workspace task=metaworld/metaworld_lowdim task.task=reach policy.mf_target_type=kmf_midpoint training.resume=false
```

批量训练使用 `--mf-target-type kmf_midpoint`。以下命令检查总计 500 个 epoch、
前 200 个纯 FM、后 300 个 FM/MF 各占一半的任务配置

```bash
python scripts/train_metaworld_batch.py --tasks reach --num-epochs 500 --mf-epochs 300 --flow-ratio 0.5 --mf-target-type kmf_midpoint --batch-name "KMFgudingmiddle_$(date +%Y%m%d)" --dry-run
```

去掉 `--dry-run` 即启动训练。可以在现有批量 JSON 中加入
`"mf_target_type": "kmf_midpoint"`，CLI 的同名参数优先。
`--flow-ratio` 表示 FM 样本比例，`--mf-epochs` 表示末尾混合训练阶段的 epoch 数。
要从第一个 epoch 使用混合训练，可令 `--mf-epochs` 等于 `--num-epochs`。

`--batch-name` 设置 `output_root` 下的实验目录名。上述 Bash 命令的结果目录形如
`data/outputs/metaworld_batch/KMFgudingmiddle_20261114/`，其中每项任务仍有独立的
`<task>_seed42/` 子目录。同日多次实验可追加 `_2`、`_3` 等序号。
启动器会拒绝复用已存在的实验目录。省略 `--batch-name` 时使用原有时间戳命名。

`policy.mf_target_type` 随训练配置保存在 checkpoint 中。推理使用已有的一步平均速度更新，
`num_inference_steps=1`。续训已有实验时使用该实验保存的目标类型，可保持实验定义一致。

回归检查覆盖解析目标与梯度、四类策略入口、纯 FM、单样本 MF、图像编码器梯度、
区间加权、真实 AMF Transformer 的 CPU 短训与断点恢复，以及批量入口参数传递。
CPU 检查验证实现与训练流程，任务成功率需要通过正式训练和环境 rollout 评估。
