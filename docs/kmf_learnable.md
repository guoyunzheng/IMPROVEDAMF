# K-MF 可学习区间切分

`policy.mf_target_type=kmf_learnable` 启用可学习切分，适用于四类 MF/AMF 的
lowdim、image 策略。`meanflow` 与 `kmf_midpoint` 分别提供原始目标和固定中点目标。
四类 YAML 默认使用 `meanflow`，可通过 Hydra 或批量启动器选择目标类型。

本实现依据论文式 7、式 8 和算法 1。2026-11-16 检查时，官方仓库包含 README
和图片，README 标注代码正在准备发布。因此以下实现细节采用本项目明确给出的约定。
[论文](https://arxiv.org/abs/2610.00864)、[官方仓库](https://github.com/IntelChina-AI/K-MF)。

生成器只接收时间端点 `(t,r)`，网络结构为
`Linear(2,128) → SiLU → Linear(128,128) → SiLU → Linear(128,1) → Sigmoid`。
MF 为每条轨迹生成一个 `lambda`。AMF 在每个动作位置的时间对上使用同一个 MLP，
输出 `[B,H]` 的 `lambda`，这是对项目向量时间表示的适配。
本实现将最后一个线性层的权重与偏置初始化为零，使初始 `lambda=0.5`。
这一初始化是本项目的实现选择，论文未给出相应约定。

给定归一化动作 `a`、噪声 `e` 和时间端点，目标按以下方式计算。
模型调用顺序沿用项目的 `model(z,t,r,cond)`。

```text
lambda = G(t,r)
c      = r + stop_gradient(lambda) * (t-r)
z_t    = (1-t)*a + t*e
z_c    = (1-c)*a + c*e
v      = e-a

D_tc = stop_gradient(JVP(model, (z_t,t,c), (v,1,0)))
D_cr = stop_gradient(JVP(model, (z_c,c,r), (v,1,0)))
D_tr = stop_gradient(JVP(model, (z_t,t,r), (v,1,0)))
D_split = (1-lambda)^2 * D_tc + lambda^2 * D_cr
target  = v - (t-r) * D_split
u       = model(z_t,t,r,cond)

delta_norm = L2_norm(flatten(D_split-D_tr))
omega      = 1 + gamma * exp(-1/(delta_norm+1e-6))
L_gen      = mean_batch(omega * mean_action_elements((stop_gradient(u)-target)^2))
L_total    = L_policy(u-stop_gradient(target)) + alpha * L_gen
```

`delta_norm` 是逐样本将动作序列的 `H,D` 两维展开后计算的未平方 L2 范数。
辅助损失的残差先按动作元素取均值，再按 batch 取均值。论文以平方 L2 范数书写式 8，
同时将目标称为 MSE；当前逐元素均值是本项目的 reduction 约定，官方代码发布后可以进一步核对。

策略损失使用项目已有的自适应 L2 归一化；AMF lowdim 可在其外施加 progressive gap
权重。生成器辅助损失直接使用上述加权均方残差，梯度经 `lambda` 的显式平方权重
及 `omega` 更新生成器。策略损失中的目标整体停止梯度，辅助损失中的策略预测与各段
JVP 停止梯度。主预测 `u` 与整段导数 `D_tr` 来自同一次 JVP，
使用同一组 dropout 掩码；两段子区间 JVP 沿 batch 维合并计算。
子区间的条件特征停止梯度，主预测保留策略网络与图像编码器的反向传播。

默认 `policy.kmf_alpha=0.1`，允许有限且非负的值。
默认 `policy.kmf_gamma_max=0.5`，允许 `[0,1]` 内的有限值。
生成器使用策略优化器的学习率，独立参数组的 weight decay 设为零。
这是本项目的优化器约定。
`gamma` 随混合 FM/MF 阶段的训练 batch 从 `-kmf_gamma_max` 线性推进到
`+kmf_gamma_max`。纯 FM 预热阶段直接回归速度并跳过生成器，预热 batch 排除在
这一进度之外。混合 batch 的辅助损失使用全部样本，包括其中 `t=r` 的 FM 样本。
这些样本的目标为 `v`，其 `omega` 仍可对生成器产生梯度，遵循论文式 8 的全 batch 定义。

在 Linux 服务器的项目根目录，以下命令使用 seed 42、500 个 epoch、前 200 个
纯 FM epoch，以及后 300 个 FM/MF 各占一半的混合 epoch。任务列表、GPU 和并发数
由批量 JSON 配置提供，命令同时启用 `k=1` 的 progressive gap 权重。

```bash
python scripts/train_metaworld_batch.py \
  --config batch_configs/metaworld_amf.json \
  --seeds 42 \
  --num-epochs 500 --mf-epochs 300 --flow-ratio 0.5 \
  --mf-target-type kmf_learnable \
  --kmf-alpha 0.1 --kmf-gamma-max 0.5 \
  --gap-weighting progressive --gap-k 1 \
  --batch-name "KMFlearnable_$(date +%Y%m%d)_seed42"
```

追加 `--dry-run` 可检查展开后的全部训练命令。结果目录位于配置的 `output_root`
下，例如 `data/outputs/metaworld_batch/KMFlearnable_20261116_seed42/`，每个任务
使用独立的 `<task>_seed42/` 子目录。`--flow-ratio` 表示 FM 样本比例。

批量 JSON 接受 `"mf_target_type": "kmf_learnable"`、`"kmf_alpha": 0.1` 和
`"kmf_gamma_max": 0.5`，同名 CLI 参数优先。单任务 Hydra 命令可使用
`policy.mf_target_type=kmf_learnable policy.kmf_alpha=0.1 policy.kmf_gamma_max=0.5`。
推理使用已有的平均速度更新，`num_inference_steps=1` 时一次前向生成动作。
