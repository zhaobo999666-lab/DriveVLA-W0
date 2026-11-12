# A53 + A54 双机 MFU 调优记录

2026-11-11，16 张 Ascend 950DT，Qwen3-VL-8B + ROSS、OpenScene/NAVSIM trainval 子集、BF16、FlashAttention-2、融合 gate/up、梯度累积 1。发布版本 `d9efa30`。训练日志和 `run-info.txt` 确认使用 **ZeRO-2**，加载 `scripts/sft/zero2.json`，没有 CPU 卸载。ZeRO-2 的单元素梯度视图兼容处理保留 NaN/Inf 检测。

## 结果与计数口径

基线和最佳档确认均为 50 步，取第 11–50 步；筛选时每档为 12 步短测，取第 6–12 步。不能用短测含热身的端到端吞吐直接和 50 步基线排序。所有用时来自关闭 profiling 的训练；Insight 轨迹是独立采集。

| 每卡 batch | 配置 | 稳态平均步时 | 骨干 useful | useful + ROSS | 骨干 + 重计算 | 骨干 + ROSS + 骨干重计算 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 12 步，完整重计算 | 8.215 s | 47.04% | 47.80% | 60.69% | 61.45% |
| 24 | 50 步基线，完整重计算 | 12.180 s | 47.59% | 48.36% | 61.40% | 62.17% |
| 27 | 12 步，完整重计算 | 13.673 s | 47.69% | 48.46% | 61.53% | 62.30% |
| 32 | 12 步，完整重计算 | 16.212 s | 47.67% | 48.44% | 61.51% | 62.27% |
| 24 | 12 步，最后 2 个文本层不重算 | 12.085 s | 47.97% | 48.74% | 61.31% | 62.07% |
| 24 | 50 步确认，最后 2 个文本层不重算 | **12.072 s** | **48.02%** | **48.79%** | 61.37% | 62.14% |

本轮最高有效值是 batch 24、最后 2/36 个文本层不重算。A53 和 A54 的 50 步确认均正常退出。同取第 11–50 步，步时从 12.180 s 降到 12.072 s（约 0.89%），端到端吞吐从 30.05 增至 30.27 样本/秒，含 ROSS 有效 MFU 增加 0.43 个百分点。优化档短测的 rank 0 峰值 HBM 为 79.5 GB，原基线为 67.2 GB；常规训练若优先显存余量仍可用原 batch 24。

以含 ROSS 的 useful 口径、固定每卡 batch 24 与 486 TFLOP/s，达到 60% 需约 **9.82 s/步**；距当前最好 50 步结果仍差约 **2.25 s/步**。计入重计算后的 62% 不能用于通过 useful MFU 验收。只修正 valid-token CE 后，最佳 50 步的含 ROSS useful 为 **46.11%**。

同一份数据用训练 collator 抽取 32 条真实样本，逐样本估算后平均。基线骨干为 **1.173843×10¹⁴ FLOPs/样本**。ROSS 额外 **1.881569×10¹² FLOPs/样本**，由本次训练形状的独立 CPU 前向 hooks 和训练乘数得到：UNet 1.270273×10¹²、冻结 VAE 编码 5.925914×10¹¹、两个条件投影 1.870406×10¹⁰。VAE 在训练中走 `no_grad`，只算一次前向。CPU 用于统计操作形状，不用其运行时间计算 MFU。

```text
useful MFU = (骨干 FLOPs [+ ROSS useful FLOPs]) × 每卡 batch ÷ (稳态平均步时 × 每卡峰值)
```

含重计算的列另加 Qwen3 文本和 ViT 的 checkpoint 前向估算，不计 ROSS 重放、优化器、通信或其他运行时额外开销。它不是 useful MFU，也不是硬件计数器实测的全部执行 FLOPs。华为云 [950DT 规格](https://support.huaweicloud.com/productdesc-aics/aics_01_0005.html) 标的是 **FP16 486 TFLOP/s**；本轮沿用 486 作为 BF16 比较分母，客户验收前须确认 BF16 峰值及 FLOPs 口径。

## MindStudio Insight：先优化哪块

对基线 16 个 rank 的第 28 步导出表取中位数：阶段 **12.30 s**、计算 **11.17 s**、通信 **0.94 s**、未被计算覆盖的通信 **0.24 s**（各 rank 0.17–0.31 s）、Free **0.88 s**。Free 还需对齐主机线程和设备流，不能全部归为 DataLoader。算子统计的 `MatMulV3` 累计 **7.57 s**（约占 65.76% 的算子累计时长），FA2 前向和反向分别 **0.98 s** 和 **0.96 s**。算子累计耗时可能重叠，不能直接相加成端到端步时。

1. 先拆解文本、ViT、重放前向和 ROSS 的 MatMul，按关键路径和输入形状排优先级。对减少重算的改动，同时确认 HBM、loss、梯度范数和稳态速度。
2. 再把 Free 区间对齐到数据准备、H2D、同步和优化器。每轮只改一项，以端到端吞吐核实收益。
3. 通信继续看长尾 rank 和扩容后的重叠率，但当前把未重叠 HCCL 全省掉也不够到 60%。不优先把 ZeRO-2 换成 ZeRO-3；若未来显存约束阻挡有效优化，再在同 batch 下实测 ZeRO-3 无卸载。

## 如何把 MFU 算得更准

- 已将 ROSS 的 UNet、冻结 VAE 和条件投影分开计数，补齐 VAE 四维注意力输入、位置参数传入的交叉注意力长度。对选择性重计算，按实际跳过的文本层数修正重放 FLOPs，并核对训练日志。
- 为与 MindSpeed 参考口径可比，主表仍按全部有效序列 token 计 LM head。真实训练开启 valid-token CE，只对监督位置投影。同一批 32 条样本平均每条 1,768.16 个有效序列 token，却只有 13.47 个监督位置。只纠正这一项，平均少计原公式中 6.552×10¹² FLOPs/样本，batch 24 基线的含 ROSS useful 由 48.36% 变为约 **45.70%**，最佳档由 48.74% 变为约 **46.06%**。这是对计算口径的审计，不代表训练变慢，也不能替换主表后再与历史 48.4% 比较。
- 32 条离线抽样用于快速复算。正式验收建议采集计时窗口每个 rank 的实际输入形状、有效标签数和步完成时间，用同窗口的总 FLOPs ÷ 总时间，同时保留 useful、含重放估算、端到端吞吐和最长 rank 步时。

## 原始产物

共享根 `/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu`。原始 50 步：`mfu-a53-a54-a5354-1612-measure`；Insight 16 rank：`mfu-a53-a54-a5354-1612-profile`；本轮扫档启动日志、ROSS 计数 JSON 和四口径表：`dispatch-tune-a5354-1645`（表文件 `mfu-comparison.md`）；最佳短测：`mfu-a53-a54-tune-a5354-gc2-b24-measure`。四口径表由 [benchmark_mfu_extended.py](../utils/benchmark_mfu_extended.py) 读取真实运行的 `trainer_state.json`、`shapes.json` 与 `run-info.txt` 生成。

需要确认最佳短测配置时，先按[双机手册](a53-a54-dual-mfu-insight.zh-CN.md)做资源和 HCCL 检查，再在 A53、A54 宿主机依次运行同一条命令（重跑改 ID）：

本次 50 步确认的共享对比表位于 `dispatch-a5354-gc2-50-1734/comparison.md`（相对于上述共享根）。

```bash
cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/mfu-dual-v2
ZB_PAIR_RUN=a5354-gc2-confirm01 ZB_JOB=measure ZB_MBS=24 ZB_FUSED_GATEUP=1 \
ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10 bash benchmark-mfu.sh
```
