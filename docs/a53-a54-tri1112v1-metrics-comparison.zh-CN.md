# A53 8 卡与 A53+A54 16 卡指标对比

实验 `tri1112v1`，2026-11-12。Qwen3-VL-8B-Instruct + ROSS，OpenScene/NAVSIM trainval，BF16，FA2，ZeRO-2，无 CPU 卸载，融合 gate/up。每卡 batch 24，梯度累积 1，最后 2 个文本层不重算。每种规模的测速和 FLOPs 计数均已完成 50 步；取第 11–50 步为稳态。

## 实测训练性能（各计算口径共用）

| 指标 | 单机 A53 · 8 卡 | 双机 A53+A54 · 16 卡 |
| --- | ---: | ---: |
| 全局 batch | 192 | 384 |
| **time_elapsed：50 步 Trainer runtime** | **657.158 s（10:57.16）** | **641.901 s（10:41.90）** |
| 稳态 40 步累计 wall time | 496.641 s | 492.567 s |
| 平均稳态 wall time（含等数据） | 12.416 s/步 | 12.314 s/步 |
| 稳态吞吐 | 15.46 样本/s | 31.18 样本/s |
| 50 步端到端吞吐 | 14.61 样本/s | 29.91 样本/s |
| **稳态 TPS（实测输入张量行数，含 padding）** | **27,427.11 token/s** | **55,306.54 token/s** |
| 稳态有效输入 TPS（按 32 条样本估算） | 27,342.59 token/s | 55,137.42 token/s |
| **HBM（框架分配峰值，仅全局 rank 0）** | **86.7 GB** | **79.5 GB** |
| Bubble（Insight 第 28 步，各 rank 中位数） | **0.000 ms** | **0.000 ms** |
| CommNotOvl（Insight 第 28 步，各 rank 中位数） | **226.555 ms** | **456.391 ms** |
| HCCL 总耗时（Insight 第 28 步，各 rank 中位数） | **882.752 ms** | **1,094.099 ms** |

- `time_elapsed` 用 Trainer 的 `train_runtime`，不是从提交 tmux 到结束的墙钟；稳态 wall time 每步取各 rank 中最慢的一个。
- TPS 的分子是该窗口所有 rank 的 `input_ids` 张量元素数，包含语言主干处理的视觉 token 和 padding，不重复加 ViT patch，也不等于 LM head 的监督行数。有效 TPS 用同一轮的抽样均值 1,768.16 token/样本推算。
- 16 卡相对 8 卡：稳态吞吐 **2.017×（线性度 100.83%）**，端到端吞吐 **2.048×（线性度 102.38%）**。一次结果不足以证明稳定超线性。

## 三种计算方式（同一实测步时）

| 计算方式 | 峰值假设 TFLOP/s/卡 | 8 卡有效 FLOPs/卡/步 | 8 卡 MFU | 8 卡 HFU | 16 卡有效 FLOPs/卡/步 | 16 卡 MFU | 16 卡 HFU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 文档一：仅投影 | 486.354 | 2.231092e15 | 36.95% | — | 2.231047e15 | 37.25% | — |
| 文档二：仅投影，重算另列 | 432 | 2.231092e15 | 41.60% | 54.59% | 2.231047e15 | 41.94% | 55.04% |
| MindSpeed：骨干 | 486 | 2.817223e15 | 46.69% | 59.67% | 2.817223e15 | 47.07% | 60.16% |
| MindSpeed：骨干 + ROSS | 486 | 2.862380e15 | 47.44% | 60.42% | 2.862380e15 | 47.83% | 60.92% |
| MindSpeed：骨干 + ROSS，修正 LM head | 486 | 2.705133e15 | 44.83% | 57.81% | 2.705133e15 | 45.20% | 58.29% |

计算口径不影响用时、TPS 或显存。文档一与文档二的有效投影分子一致，峰值不同；MindSpeed 统计范围更宽。HFU 为含重算估算，不能用于有效 MFU 60% 的验收。

## 通信和 Bubble 补采

`tri1112v1` 的正式测速轮关闭了 profiling。表中的通信数据来自独立的同配置补采，空缺不代表零。三项分别使用 MindStudio Insight 的 `Bubble`、`Communication(Not Overlapped)` 和每 rank 内 HCCL 操作的 `Elapse Time(ms)` 累计；另外保留时间线上通信的总覆盖时长 `Communication`。操作累计时间可能重叠，不能加到步时上。Bubble 也不等于一般的设备空闲（Free）。

A53 单机和 A53+A54 双机补采各完成 32/32 步，8 和 16 份 rank 数据齐全。第 28 步的 CommNotOvl 范围由 **134.060–359.675 ms** 增至 **370.630–903.866 ms**；HCCL 操作累计范围由 **451.022–1,119.968 ms** 增至 **844.028–1,539.282 ms**；Stage 中位数由 **12,211.061 ms** 增至 **12,441.730 ms**。这里比较的是带采集的同一步，正式线性度仍用上面的 50 步独立测速。第一次双机补采因 A54 预检时无可确认的 8 张空闲卡而未进入训练；第二次配对成功。

已有的 **11 月 11 日双机完整重算**轨迹只供参考：16 个 rank 在第 28 步的中位数为 **Bubble 0.000 ms、CommNotOvl 241.238 ms、HCCL 操作累计 941.784 ms**。这份采集和本轮末两层不重算的配置不同，不参与单机与双机的正式对比。

同配置独立补采使用相同训练快照、MBS 24、末两层不重算，各跑 32 步，按同一采集位置核对 8 和 16 个 rank。补采的步时不替换上面的正式测速。8 卡原始汇总在 `insight-gc2-01/single-profile.md`；完整配对汇总位于：

```text
/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/three-methods-tri1112v1/insight-gc2-02/paired-profile.md
```

来源：[完整基准报告](a53-a54-tri1112v1-performance-report.zh-CN.md)、[旧 Insight 分析](a53-a54-local-insight-analysis.zh-CN.md)、[补采汇总脚本](../utils/report_mfu_profiler.py)。训练原始文件在共享盘对应 `mfu-a53-tri1112v1-{measure,audit}` 和 `mfu-a53-a54-tri1112v1-{measure,audit}` 目录。
