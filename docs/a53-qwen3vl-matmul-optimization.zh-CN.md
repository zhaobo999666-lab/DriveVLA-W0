# Qwen3-VL + ROSS：矩阵乘和 MLP 怎么省时间

数据来自 A53 8 卡、MBS 24、文本末两层不重算的 Insight 第 28 步。正式测速的稳态步时是 12.416 秒；带采集的这一阶段是 12.211 秒。算子累计时间不是整步关键路径。

## 矩阵乘为什么久

单卡一次处理约 42,528 个文本主干位置。融合 gate/up 把 4,096 维投影到 `2 × 12,288 = 24,576` 维。一次乘法约 `2 × 42,528 × 4,096 × 24,576 ≈ 8.56e12 FLOPs`，rank 0 实际约 **20 ms**。36 个文本层都做正常前向，34 个 checkpoint 层在反向时再做一次，共 **70 次**、仅 gate/up 前向累计 **1.398 秒**。down projection、输入梯度和权重梯度也各自有矩阵乘。合在一起，8 个 rank 的文本 MLP 中位累计时间是 **4.245 秒**，其中 **1.018 秒**是已识别的重算前向。

这类 gate/up GEMM 按形状和实际设备时间推算约 **429 TFLOP/s**，说明它花时间主要是因为工作量大、次数多。不要把这个单算子吞吐当作 MFU。已经开启 gate/up 融合；本步全部 `ConcatD` 累计约 **24.6 ms**，这个值包含其他拼接，因此把两个权重做成永久合并的参数不是当前优先项。

## 第一项：FA2 去 padding 的元数据每批只准备一次

同一批的文本 mask 会传给 36 个文本层，并用于 34 个层的重算。rank 0 实际记录了 70 次同形状的 `nonzero`。原实现会在每层重复准备有效索引、累计序列长度和最大长度；这里有 `.item()`，可能产生主机和 NPU 同步。两次轨迹都看到过文本前向附近约 0.45 秒的设备空档，但还不能把整段空档归因给这一个函数。

新增 `ZB_FA2_UNPAD_CACHE=1`，默认关闭。只在当前训练 microbatch 的前向、反向和重算之间复用同一个 mask 的元数据。mask 更改后重新计算；本步结束立即清除。Q/K/V 的 gather、FlashAttention 算法、位置编码、padding mask 和 MLP 算法保持原样。另有 `ZB_FA2_UNPAD_MARKERS=1`，仅用于独立采集时标记真正进行元数据准备的调用。

上线验证：A53 短测 ID 为 `fa2ab1112-01`，两轮各 12 步，取第 6–12 步。两轮均使用独立共享快照 `tri-mfu-1112-fa2-v1`，只记录调用计数，不开 profiler 标记。训练完成后生成 `/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/fa2ab1112-01/comparison.md`，另有每 rank 的 `fa2-unpad` 与 `peak-memory` 数据。

1. 用同一独立快照、A53 8 卡、MBS 24、ZeRO-2、FA2、末两层不重算。先用 12 步短测，baseline `ZB_FA2_UNPAD_CACHE=0`，试验 `ZB_FA2_UNPAD_CACHE=1`，各用新 run ID、相同热身和 seed。
2. 查日志 `[fa2-unpad]`，确认命中量，再对比 loss、梯度范数、全 rank 峰值 HBM 和稳态步时。CPU 等价性测试不代表 NPU 速度已经改善。
3. 两轮都关闭标记与 profiling，跑同配置 50 步比较稳态与端到端吞吐。另跑相同位置的 Insight，使用 `ZB_FA2_UNPAD_MARKERS=1`，比较元数据调用数和 Free 空档。不要用带标记的步时当作正式成绩。

## 第二项：减少 MLP 重算，同时管住显存

跳过更多 checkpoint 能省矩阵乘，但不能直接关掉全层重算。仅一次 gate/up 的 BF16 输出就约 **2.09 GB**（42,528 × 24,576 × 2 字节）；不能把所有 34 层这样的输出都常驻 HBM。当前 8 卡 rank 0 峰值为 86.7 GB，之前 16 卡试过跳过末 4 层，已有卡接近满显存。

- 先保持 GC2，按模块拆分重算。测试选择性保存少数层的昂贵中间结果，并保留其他层重算；用实际全 rank HBM 定预算。不要把 1.018 秒算子累计全当作可省的墙钟时间。
- 如果允许改变输入规格，图像 resize 是另一条路线：两张图各约 3,432 个 ViT patch，经 2×2 合并共约 1,716 个文本主干视觉位置。减少视觉 token 同时减少 ViT 和文本 MLP 的工作量，但需独立验证模型效果和重新统计 FLOPs。
- 未重叠通信约 0.23 秒。现有单机到双机吞吐已接近翻倍；本轮不优先换 ZeRO 或为同一大矩阵乘再换融合方式。

依据：[A53 步时拆解](a53-gc2-step-time-breakdown.zh-CN.md)、[单机与双机正式对比](a53-a54-tri1112v1-performance-report.zh-CN.md)、[FA2 缓存实现](../utils/fa2_unpad_cache.py)、[基准入口](../scripts/a5/benchmark-mfu.sh)。
