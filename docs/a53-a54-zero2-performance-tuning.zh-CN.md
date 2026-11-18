# A53+A54：基于 Insight 的 ZeRO-2 性能调优实施说明

更新时间：2026-11-13。

## 1. 为什么先优化通信调度

上一轮 `fa2dual1113-01` 的无 Profiler 稳态步时为 **12.2517 s**，有效 MFU 为
**48.07%**，含重算 HFU 为 **61.23%**。Insight 的关键路径中，计算约占
88.18%，未重叠通信约占 5.00%，普通设备空档约占 6.90%。每 rank、每步约有
398 次 AllReduce，其累计时间的约 85.8% 是等待 participant 到达，而不是数据
在链路上传输。

因此第一轮不更换模型、不减少视觉 token，也不改变训练语义，先调整 ZeRO-2
梯度 bucket、通信重叠与 HCCL 确定性。它们可能压缩同步长尾，风险低且容易回退。
文本 MLP 的大 GEMM 已约 429 TFLOP/s，单纯替换同一矩阵乘调用不太可能省下目标
所需的 19.9%。

## 2. 已完成的代码改造

- `ZB_ZERO2_BUCKET_SIZE=auto|正整数`：单位是元素数。Qwen3-VL 隐藏维度为
  4096，Transformers 将 `auto` 解析为 `4096² = 16,777,216`。
- `ZB_ZERO2_OVERLAP_COMM=0|1`：控制 ZeRO-2 的 `overlap_comm`，默认 1。
- `ZB_HCCL_DETERMINISTIC=0|1`：控制 HCCL 确定性，默认 1，保持历史行为。
- 每次运行生成独立 `deepspeed-zero2.json`，不修改仓库默认 `zero2.json`。
- 三项配置进入 `run-info.txt`、双机签名和 workload 签名；两节点不同会拒绝启动。
- 保存 16 个 rank 的关键路径步时和峰值 HBM，自动拒绝训练未完成、loss 非有限或
  配置与记录不一致的结果。

共享盘部署为：

```text
/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/tri-mfu-1113-zero2-ab-v1
```

基础提交为 `bf729327b24b687bb910153dfe5d74c34dbb1b69`。该目录有独立
`.mfu-sha256` 清单，未覆盖上一轮 FA2 快照。

## 3. 第一轮短测矩阵

所有组固定 Qwen3-VL-8B + ROSS、OpenScene/NAVSIM trainval、16 卡、BF16、
FA2、ZeRO-2、MBS 24、GBS 384、GC2、FA2 metadata cache、30 步和热身 10 步。
每次只改变一个变量：

| 组 | bucket 元素 | overlap | deterministic | 作用 |
|---|---:|:---:|:---:|---|
| base | auto = 16,777,216 | 1 | 1 | 同日基线 |
| bucket2x | 33,554,432 | 1 | 1 | 检查减少小 bucket 同步 |
| bucket4x | 67,108,864 | 1 | 1 | 检查更大 bucket；同时观察 HBM |
| detoff | auto | 1 | 0 | 单独估计确定性开销 |
| overlapoff | auto | 0 | 1 | 诊断通信重叠是否实际有效，预计可能变慢 |

在 A53、A54 各开一个 tmux，然后使用相同的运行 ID：

```bash
tmux new -s zhaobo-zero2-ab
bash /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/tri-mfu-1113-zero2-ab-v1/run-zero2-ab.sh zero2ab-01
```

报告会生成在：

```text
/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/zero2-ab-zero2ab-01/comparison.md
```

表中同时给稳态均值/P50/P95、samples/s、全 rank 最大 HBM、末次 loss，以及按上一
轮相同 FLOPs 分子缩放的投影有效 MFU/HFU。投影值只用于筛选；胜出组仍需 50 步
正式测速和独立 projection audit。HFU 含 checkpoint 重算，不能用于 60% 有效 MFU
验收。

## 4. 胜出后的正式验证

将下例的 bucket、overlap、deterministic 换成短测胜出值，在两台机器同时执行：

```bash
export ZB_PAIR_RUN=zero2best-50 ZB_JOB=measure
export ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_FAST_ZERO_STAGE=2
export ZB_GC_SKIP_TEXT_LAST=2 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10
export ZB_FA2_UNPAD_CACHE=1 ZB_ZERO2_BUCKET_SIZE=33554432
export ZB_ZERO2_OVERLAP_COMM=1 ZB_HCCL_DETERMINISTIC=1
bash /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/tri-mfu-1113-zero2-ab-v1/benchmark-mfu.sh
```

正式 50 步至少重复两次。只有步时改善超过同日重复波动、loss 有限、16 个 rank
均完整退出且 HBM 有安全余量，才接受为新基线。随后用同配置 `ZB_JOB=audit`
计算准确的有效 MFU/HFU；再独立采集 Profiler，检查 AllReduce 次数、
CommNotOvl、P95 长尾和 Free 是否确实下降。Profiler 步时不用于正式 MFU。

## 5. 计算侧下一步

通信最优值确定后，单独比较 GC2 与 GC3。GC3 预计每步只省约 30 ms，因此是小收益；
必须观察 16 个 rank 的最大 HBM。历史 GC4 曾达到 98,267/98,304 MB，不能直接
作为长跑配置。若完全消除未重叠通信和 Free，理论有效 MFU仍约 54.8%；要达到
60%，还需要从计算路径省约 0.94 s。后续的大收益方向只能是选择性保存昂贵的
MLP 中间结果、冻结部分 ViT 或减少视觉 token，但后两项会改变训练语义，需要
单独验证驾驶指标。

## 6. 当前机器状态

2026-11-13 15:25（Asia/Shanghai）检查：A54 8 卡空闲；A53 8 卡由容器
`lh-vllm` 占用，进程为 `VLLMWorker_DP`，每卡约占 92 GB。该任务不是本训练容器，
因此没有停止它，也没有启动双机短测。A53 释放后再执行第 3 节命令。
