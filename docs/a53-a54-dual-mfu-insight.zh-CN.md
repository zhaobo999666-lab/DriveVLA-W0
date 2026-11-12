# A53 + A54 双机 MFU 与 MindStudio Insight

2026-11-11，16:00 后按实际资源分配启动。A53 是 rank 0、会合地址 `10.0.16.231:29555`；A54 是 rank 1、地址 `10.0.16.118`。两机各 8 卡。两台都在宿主机执行下面相同命令，**每个阶段先 A53、随即 A54**；两端参数必须一致。重复试跑时换新的 ID。

1. 先查 `npu-smi info`，两端必须各有 8 张空闲卡。15:36 检查两端卡均空闲，容器就绪，代码 `d9efa30` 一致，私有 `/dev/shm` 均为 96 GiB。这是快照，16 点开跑前复查。

2. 验收 16 卡 HCCL。两端都要成功，归约结果为 136。

   ```bash
   cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56
   export ZB_NODE_RANK=0  # A53；在 A54 把这里改成 1
   ZB_MASTER_ADDR=10.0.16.231 ZB_STRICT_RELEASE=1 bash start-training.sh check a5354-hccl-01 fast 2 qwen3vl 2
   ```

3. 同配置冒烟。用单机已验证过的性能档：Qwen3-VL-8B + ROSS、OpenScene/NAVSIM trainval 子集（61,898 条训练样本）、FA2、修复版 ZeRO-2、融合 gate/up、BF16、每卡 batch 24、梯度累积 1、全局 batch 384。若显存不足，两个节点一起把 MBS 降到 16，再用新 ID 冒烟。

   ```bash
   cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/mfu-dual-v1
   ZB_PAIR_RUN=a5354-01 ZB_JOB=smoke ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_MFU_STEPS=3 ZB_MFU_WARMUP=0 bash benchmark-mfu.sh
   ```

4. 正式计时（先确认两端冒烟成功）。不采集、不存 checkpoint；取第 11–50 步。rank 0 计算 16 卡 MFU，报告及两节点日志存共享盘。

   ```bash
   ZB_PAIR_RUN=a5354-01 ZB_JOB=measure ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10 bash benchmark-mfu.sh
   ```

5. 独立采集。同样的 batch 和优化开关，跑 32 步，Level 1 的现有采集计划在跳过前 26 步、等待和热身后采 1 步。正式 MFU 不用这轮时间。两机各自的输出目录避免相同文件名碰撞。

   ```bash
   ZB_PAIR_RUN=a5354-01 ZB_JOB=profile ZB_MBS=24 ZB_FUSED_GATEUP=1 ZB_MFU_STEPS=32 ZB_MFU_WARMUP=10 bash benchmark-mfu.sh
   ```

共享输出根：`/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu`。正式报告在 `mfu-a53-a54-a5354-01-measure/summary.txt`。采集输出在 `mfu-a53-a54-a5354-01-profile/node-rank-{0,1}/train-fast/navsim/npu_profile_level1`。先检查两个节点采集目录齐全、识别出 16 个全局 rank，再用 [MindStudio Insight 导入性能数据](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0059.html)。保留整棵采集目录及元数据；按实际生成的格式导入 `ASCEND_PROFILER_OUTPUT` 中的时间线和通信数据，若导出的是数据库则保留 `analysis.db` 和各 rank 的数据库。参见[昇腾 PyTorch Profiler 产物说明](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T%26ITools/Profiling/atlasprofiling_16_0006.html)。

## 60% 验收和瓶颈定位

先确认客户的 60% 指 **useful 还是含重计算、是否含 ROSS、峰值按什么 BF16 规格、步时是否含数据等待**。当前报告同时给骨干 useful 和端到端口径，分母暂用每卡 486 TFLOP/s，不能拿 executed MFU 代替 useful MFU 过线。沿用当前骨干口径与每样本约 1.174e14 FLOPs，MBS 24 要到 60%，稳态平均步时需 **≤9.66 秒**；A54 单机已测 **12.188 秒**，意味着即使不计跨机新增开销，也要把步时压低约 **21%**。先出双机实测再判断差距。

| Insight 上优先看 | 证据 | 下一轮实验 |
| --- | --- | --- |
| HCCL 与计算的重叠 | 16 rank 时间线对齐，找没被计算覆盖的 ReduceScatter/AllGather 和慢 rank；不要把已重叠通信再加到步时里 | 固定 MBS 和数据，和 A54 单机比较；通信是主因才试桶大小、重叠或核查跨机链路 |
| Host/DataLoader、H2D | NPU 空洞是否对应 CPU 备数据、等待队列或 H2D；对照正式跑的端到端吞吐与计算步时差 | 一次只改 worker、预取或数据位置的一项，留意两个节点数据读取是否失衡 |
| 计算和优化器 | Operator/Timeline 对照 MatMul、FA2、ROSS、优化器以及重计算区间，按关键路径耗时排序 | 先优化占比最大的算子或路径，再跑同配置 50 步验证；检查 loss 和 HBM |

分析交付应记录每项候选瓶颈的毫秒数、是否位于关键路径、优化前后 MFU 与吞吐。同一 MBS 下若双机单步时间不变，整体吞吐应接近单机的两倍；也可用双机吞吐 ÷（2 × 同配置单机吞吐）衡量扩展效率。
