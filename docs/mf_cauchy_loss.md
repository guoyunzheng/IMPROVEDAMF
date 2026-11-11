# MF 固定尺度对数损失

MF 和 AMF 的 lowdim、image 配置均新增 `policy.mf_loss_type` 和
`policy.mf_loss_c`。默认 `mf_loss_type=adaptive_l2`，保留现有基线；设为
`cauchy` 时，仅 MF 样本改用固定尺度对数损失。FM 样本及纯 FM 预热阶段
仍使用原有自适应 L2，采样比例、JVP 目标和推理过程不变。

```bash
python train.py --config-name=train_adaptive_mean_flow_transformer_lowdim_workspace task=metaworld/metaworld_lowdim task.task=pick-out-of-hole policy.mf_loss_type=cauchy policy.mf_loss_c=0.01
```

对每条轨迹先在动作时间维和动作通道维计算均方残差 `E`，再计算
`log1p(E / c)`，最后对完整 batch 求平均。固定 `c` 时，该表达式与
`log(E + c)` 只差常数，梯度相同；没有额外的可学习尺度网络，也不会再叠加
原 MF 自适应 L2 权重。`c` 必须是有限正数，默认 `1e-3`；示例中的 `0.01`
仅展示配置方式，需要通过任务实验选择。

AMF lowdim 可叠加现有 progressive gap 加权：

```bash
python train.py --config-name=train_adaptive_mean_flow_transformer_lowdim_workspace task=metaworld/metaworld_lowdim task.task=pick-out-of-hole policy.mf_loss_type=cauchy policy.mf_loss_c=0.01 training.progressive_gap_weighting.enabled=true
```

组合方式为 `gap_weight * log1p(E / c)`，权重位于对数外，FM 权重保持 1。
该分支保留 `fm_mse`、`mf_mse` 和原始 `*_adaptive_loss` 诊断，并新增
`*_objective_loss` 记录实际采用的、未乘 gap 权重的损失；验证时使用对应
`val_` 前缀。progressive gap 仍只支持 AMF lowdim。

新参数保存在检查点配置中；旧配置缺少这些参数时默认使用原基线。
比较实验应使用相同初始化、数据与采样设置，分别指定不同的输出目录，
避免默认续训误加载其他实验。两种损失的数值不可直接比较，建议比较原始
残差 MSE、任务成功率和梯度稳定性。

## 验证

```bash
python -m unittest discover -s tests -p 'test_mean_flow_*.py' -v
python -m unittest discover -s tests -p 'test_progressive_gap*.py' -v
```

覆盖逐轨迹损失及解析梯度、仅 MF 受 c 影响、FM 与默认基线兼容、采样随机流
不变、显式 FM 标记、纯 FM 跳过 JVP、纯 MF 与混合 batch、半精度大残差、
gap 外部加权、诊断聚合，以及真实 Transformer 的 CPU 短训、EMA 和检查点恢复。
本地本轮共运行 61 项，60 项通过，1 项 CUDA 测试因安装的 PyTorch 不支持
本机显卡架构而跳过。测试中的数据集和环境使用轻量适配器，未进行完整
MetaWorld/PushT 闭环训练；这些验证不代表任务成功率已经提高。
