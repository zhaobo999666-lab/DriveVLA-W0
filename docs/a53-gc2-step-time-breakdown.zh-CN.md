# A53 单机 GC2：12 秒步时拆解

基准 `tri1112v1` 的稳态含等待步时：A53 8 卡 **12.416 秒**，A53+A54 16 卡 **12.314 秒**。独立 Insight 补采 `mfu-a53-insight-gc2-01-profile` 采用相同模型、MBS 24、FA2、ZeRO-2、融合 gate/up、末两个文本层不重算，采集第 28 步。以下分解来自这次 **8 卡** 补采，取 8 个 rank 的中位数；不能把这一条轨迹当成 40 步平均或同配置 16 卡通信数据。

| Insight 项目 | 时间 | 相对阶段步时 |
| --- | ---: | ---: |
| Stage | 12.211 s | 100% |
| Computing | 11.100 s | 90.9% |
| CommNotOvl | 0.227 s | 1.9% |
| Free | 0.880 s | 7.2% |
| Preparing | 0.00084 s | <0.01% |

通信总耗时中位数 **0.883 s**，多数和计算同时发生；关键路径上真正未被覆盖的只有约 **0.227 s**。Bubble 为 0 不表示没有空档：Bubble 是流水线气泡字段，普通设备空闲归 Free。即便把这一步的未重叠通信和 Free 全部移掉，也只省约 **1.11 s**。按含 ROSS 的历史口径，双机距有效 MFU 60% 需要约 **2.50 s/步** 的缩减；因此要同时优化计算路径。

## 算子耗时

| 算子或形状组 | 每卡累计时间中位数 | 解释 |
| --- | ---: | --- |
| MatMulV3 | 7.495 s | 约占算子累计时长 64.5% |
| 其中：文本 MLP | 4.245 s | gate/up、down 的前向、反向和重算 |
| 其中：文本 MLP 重算前向 | 1.018 s | 每卡 34 层 × 2 个矩阵乘，关联到 CheckpointFunctionBackward 内的实际前向形状 |
| ViT MLP + QKV 投影 | 0.947 + 0.358 s | 两组不同的矩阵乘，已包含在 MatMulV3 |
| FA2 前向 + 反向 | 0.979 + 0.960 s | 设备算子总计；尚未按文本、视觉和重算进一步拆分 |

算子累计时间可能重叠，表中“其中”项不能重复加到步时。以 rank 0 为例，融合 gate/up 的输入为 `[42528,4096]` 和权重 `[24576,4096]`，原始前向 36 次、重算前向 34 次，总耗时 1.398 s。按矩阵乘形状推算约 **429 TFLOP/s**，说明这个大算子虽然花时间多，但不是单次吞吐特别低的证据。要省时，先考虑少做重复前向，不能把全部重算算子时间当成保证可省的墙钟时间。

模型每个样本处理两张图，抽样每张约 3,432 个 merger 前 patch，语言主干每样本约 1,768 个有效 token，另有 ROSS 分支。Emu3 文档的 6.85 s 属于另一套模型和 batch，不能直接当成此任务的预期步时。历史含 ROSS 有效 MFU 和投影口径的百分比也各自使用不同统计范围。

## 设备空档定位

- 各 rank 最长 Free 区间的中位数为 **0.445 s**。rank 0 的空档在步内 **1.984–2.443 s**，处在 checkpoint 文本层前向。其间一条 Host `aten::flatten` 调用跨约 **0.250 s**。旧的完整重算采集也见过同类型空档。
- `aten::nonzero` 的 Host 区间合计约 **5.389 s**，但只约 **8.6 ms** 和设备 Free 相交，NPU `NonZero` 算子本身约 **2.9 ms**；不能把 Host 上显示的 5 秒认定为可省时间。DataLoader `__next__` 在这个采集步中位数约 **0.114 ms**，没有显示长时间等数据。
- `modeling_flash_attention_utils.py` 的 `_get_unpad_data` 有 `nonzero(...).flatten()` 和 `.item()`，decoder 各层收到相同 batch 的 mask。rank 0 记录了 **70 次** 输入为 `[42528]` 的 `aten::nonzero`，数量与 36 次原始文本前向加 34 次重算吻合。可优先给去 padding 元数据准备、索引 gather 和 FA2 下发加分段标记，验证重复准备或同步是否造成空档；当前未采调用栈，不能认定 flatten 是根因。

## 建议先做什么

1. 固定 MBS 24、ZeRO-2 和数据，给当前 GC2 的文本层加细粒度标记，验证 FA2 前的空档。若确认同一 batch 重复算相同索引，试把可复用的索引和序列边界按 batch 预计算；先核对 loss、梯度和注意力 mask 一致性。
2. 同时测选择性重算：优先保留计算昂贵的结果，重算便宜部分。单机 rank 0 峰值 HBM **86.7 GB**；此前 16 卡跳过末四层已有卡接近 96 GB，先查每卡实际显存余量。
3. 同配置 16 卡 Insight 补齐后对照慢 rank、通信重叠和空档。现有 8→16 卡吞吐接近翻倍，先不把换 ZeRO 或调 HCCL 当作主要优化。

来源：`/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/mfu-a53-insight-gc2-01-profile` 的 8 份 `step_trace_time.csv`、`op_statistic.csv` 和各 rank SQLite 数据库；[8/16 卡正式性能报告](a53-a54-tri1112v1-performance-report.zh-CN.md)；[上一轮完整重算 Insight 分析](a53-a54-local-insight-analysis.zh-CN.md)。
