# DriveVLA-W0：A53 + A54 本地 MindStudio Insight 分析

**主要耗时在模型计算。优先减少文本层前向重算，并拆解 FA2 和视觉层耗时。** 基线第 28 步的 16 卡中位数是 12.301 秒，未被计算覆盖的通信仅 0.241 秒。最快的 50 步训练为 12.072 秒/步、48.79% 有效 MFU（含 ROSS）。按目前的 FLOPs 估算和每卡 486 TFLOP/s 分母，60% 需要约 9.82 秒/步，仍需省约 2.25 秒。

## 分析范围

- 采集：`a5354-1612-profile`，两机 16 卡，Qwen3-VL-8B + ROSS，BF16，FA2，ZeRO-2 无卸载，MBS 24 / GBS 384，融合 gate/up，完整激活重算。Level 1 采了第 28 步。
- 正式 MFU：关闭采集，取 50 步训练的第 11–50 步。最快档跳过最后两个文本层的重算；它和基线使用相同 batch，**目前只有基线有此采集**。
- 本地 Insight 26.1.0 已导入 `D:\OTT9\DriveVLA-W0\artifacts\insight-a5354-1612\import-db`，解析成功并生成 `cluster_analysis_output/cluster_analysis.db`，包含全局 rank 0–15。按 [Insight 26.1.0 导入示例](https://www.hiascend.com/document/detail/en/mindstudio/2610/visualization_tool/MindStudioInsight/docs/en/quick_start/system_tuning_quick_start.md)，导入直接包含采集目录的上一级。`import-db` 将 16 个 `*_ascend_pt` 整理成同级；每个目录保留数据库和相关元数据。[Insight 双机时间线截图](../artifacts/insight-a5354-1612/insight-dual-timeline.png)

## 第 28 步：卡在哪里

以下取 16 个 rank 的中位数。百分比和区间只用于定位，不把重叠通信和算子累计耗时叠加到步时。

| 项目 | 耗时 | 解读 |
| --- | ---: | --- |
| Step Stage | 12.301 s | 各 rank 12.300–12.302 s |
| NPU 计算 | 11.171 s | 约占阶段的 91% |
| HCCL 未重叠 | 0.241 s | 范围 0.174–0.306 s；总通信 0.942 s，约四分之三已被计算覆盖 |
| NPU Free | 0.882 s | 需按 Timeline 继续定位同步、算子下发、H2D 和步末间隙 |
| DataLoader `__next__` | 0.283 ms | 这个采集步没有明显等待 DataLoader 的证据 |

在 Host 时间线上，`aten::nonzero` 的区间并集有 5.483 秒，但只有约 **5.6 毫秒**与 Free 相交，NPU 的 `NonZero` 本身累计约 **2.9 毫秒**。长 Host 区间与已在执行的设备工作大量重叠；不能按 5.48 秒估算优化收益，也不能把 Free 全算给 DataLoader。

值得优先复查的空档：每个 rank 最长的一段 Free 的中位数为 **0.454 秒**。rank 0 上位于步内 **1.596–2.050 秒**，落在一个接收 `[24,1773,4096]` 的前向 `CheckpointFunction` 范围内。其间一段 `aten::flatten` Host 调用跨度约 262 ms。该调用没有调用栈，且 Host 时间可能包含同步或采集开销，不能据此认定 flatten 算子是根因。对照下一次采集时，先给这一文本层的 mask、序列索引准备和 FA2 下发加标记。

## 算子拆解

| 设备算子或形状组 | 累计时间（每卡中位数） | 证据 |
| --- | ---: | --- |
| `MatMulV3` | 7.573 s | 约占全部设备算子累计时间的 65.8% |
| 文本 MLP 矩阵乘 | 4.304 s | 以 4096、12288、24576 等输入维度识别；包含前向、反向及重算 |
| 其中：文本 MLP 原始前向 | 1.077 s | 每 rank 36 层 × 2 个矩阵乘 |
| 其中：文本 MLP 重算前向 | 1.078 s | 设备任务关联到 Host `CheckpointFunctionBackward` 内对应的前向形状；每 rank 72 次 |
| 视觉 MLP 矩阵乘 | 0.947 s | 按 1152 与 4304 的形状识别 |
| 视觉 QKV 投影矩阵乘 | 0.358 s | 按 1152 与 3456 的形状识别 |
| FA2 前向 / 反向 | 0.981 / 0.962 s | Profiler 算子汇总，包含采集到的所有对应调用 |

例如 rank 0 的文本 MLP 前向，融合 gate/up 的输入形状为 `[42552,4096] × [24576,4096]ᵀ`，down projection 为 `[42552,12288] × [4096,12288]ᵀ`。两种分别执行 72 次，累计 1.438 和 0.718 秒。按 `2×M×N×K ÷ 设备算子时间` 粗算，两者均约 429 TFLOP/s。这是**单类矩阵乘按形状推算的吞吐**，不等于整步 MFU，也不是 PMU 实测的 BF16 峰值。继续提高这些大矩阵乘的单次吞吐，空间可能小于少做重复的前向。

## 建议下一轮

1. **固定 MBS 24、数据和 ZeRO-2。** 给最快档补同一位置的 Level 1 采集。正式 A/B 已看到跳过最后两层重算省约 108 ms/步，但 rank 0 的短测峰值 HBM 从 67.2 增至 79.5 GB。先找“保留代价最高的中间结果、只重算便宜部分”的策略，再用同配置 50 步验证步时、HBM、loss 和梯度范数。1.078 秒是本步文本 MLP 重算的算子累计量，不能承诺都能省下。
2. **给文本、ViT 和 ROSS 标记子模块范围。** 按阶段核查 FA2 前后向、视觉重算、ROSS 和各 rank 的最大空档。找出哪些操作在关键路径，才决定调序列 packing、融合还是同步。
3. **通信放后面。** 先保留 ZeRO-2，把未重叠通信当作约 0.24 秒量级的优化机会。只有显存明确阻挡更大的计算收益时，再做同 batch 的 ZeRO-3 对照。

客户验收建议锁定 BF16 峰值以及 LM head 的有效标签计数方式；48.79% 使用历史比较口径，只修正 valid-token CE 后为 46.11%。含重算的 62.14% 不能当作 useful MFU 达标。

复查：[16 rank 明细 CSV](../artifacts/insight-a5354-1612/step28-ranks.csv) · [SQLite 复算脚本](../artifacts/insight-a5354-1612/analyze_local.py) · [50 步 MFU 对照](a53-a54-mfu-tuning-results.zh-CN.md)。
