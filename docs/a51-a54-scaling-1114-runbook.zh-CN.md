# A51–A54 单机、双机、四机性能测试运行手册

运行 ID：`scale1114-03`  
共享快照：`/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/scaling-1114-v1`  
共享结果：`/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/scaling-scale1114-03/`

## 1. 环境

四台机器使用相同的 `zhaobo` 隔离环境：

| 项目 | 值 |
|---|---|
| 镜像 | `zhaobo/drivevla-w0:a55-cann9.2-torch2.11-py312-v1` |
| 镜像 ID | `sha256:9ec7bfba7014358ffc93f7e65fa582f9395e5732fcbb6368cf63d2d0d43d5d22` |
| 代码提交 | `bf729327b24b687bb910153dfe5d74c34dbb1b69` |
| 容器 | `zhaobo-drivevla-a51` 至 `zhaobo-drivevla-a54` |
| 网络 | host |
| `/dev/shm` | 64 GiB |
| Python | `/opt/zhaobo/train` |
| Transformers | vendored 4.56.0 |
| tokenizers | 0.22.2 |
| CANN | 9.2.0 |
| PyTorch NPU | 2.11.0 |
| DeepSpeed | 0.19.7 |

模型、数据和结果都放在共享目录：

```text
/mnt/sfs_turbo/zhaobo/drivevla-w0/models
/mnt/sfs_turbo/zhaobo/drivevla-w0/data
/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts
```

A51 的本地工作目录是 `/data1/zhaobo/drivevla-w0-a51`。A52–A54 使用各自同名的 `/data1/zhaobo/drivevla-w0-a52` 至 `-a54`。

## 2. 固定实验配置

| 项目 | 值 |
|---|---|
| 模型 | Qwen3-VL-8B-Instruct + ROSS |
| 辅助模型 | SD1.5 UNet，冻结 VAE |
| 数据 | OpenScene/NAVSIM trainval |
| 精度 | BF16 |
| Attention | FlashAttention-2 |
| 并行 | ZeRO-2，无 CPU offload |
| MBS | 24 |
| GBS | 8/16/32 卡分别为 192/384/768 |
| 梯度累积 | 1 |
| 激活重算 | 文本最后 2/36 层不重算 |
| FA2 metadata cache | 开启 |
| ZeRO bucket | auto |
| overlap_comm | 开启 |
| HCCL deterministic | 开启 |
| 测速步数 | 50；第 11–50 步为稳态窗口 |
| seed | 42 |

单机、双机和四机只改变节点数，其他配置保持相同。

## 3. 启动方式

在 A51–A54 四台机器上分别启动同名 tmux 会话，传入同一个运行 ID：

```bash
tmux new -s zhaobo-scale1114-03
bash /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/scaling-1114-v1/run-scaling-a51-a54.sh scale1114-03
```

A51 先执行单机，A51+A52 随后执行双机。A53、A54 会持续检查本机 8 张 NPU；
连续 4 次确认空闲、四台机器的就绪心跳在 45 秒内时，才启动四机 32 卡阶段。
外部容器占用 NPU 时不会抢占或终止其进程。

## 4. 每阶段执行的轮次

每个规模依次执行：

1. `measure`：关闭 Profiler 和 FLOPs 计数，用于正式墙钟、TPS、MFU/HFU 和 HBM。
2. `audit`：独立捕获投影 FLOPs，用于审计三套 MFU/HFU 计算口径。
3. `profile`：50 步 NPU/CPU Profiler，用于 Insight Step Trace、通信、Free、Preparing 和 HCCL。

Profiler 轮会放慢训练，其中的步时只用于定位耗时，不替换 `measure` 的正式速度。

## 5. Step Trace 口径

典型关系是：

```text
Stage(单步 Step Trace)
  = Computing
  + Communication(Not Overlapped)
  + Free
```

同事记录：

```text
Stage = 13,366 ms
Computing = 12,761 ms
Communication(Not Overlapped) = 242 ms
Free = 363 ms
Preparing = 41 ms
12,761 + 242 + 363 = 13,366 ms
```

因此 **13,366 ms 就是该采集步的端到端单步耗时**。Preparing 不能再加到
Stage 中；该记录缺少模型、MBS、卡数、GC、attention 和代码版本，只能作为
Step Trace 量级参考，不能直接比较 MFU。

## 6. 指标定义

- `time_elapsed`：50 步 Trainer runtime，不含模型加载和启动时间。
- `Step wall`：相邻优化器步之间，每步取所有 rank 最慢者，再对第 11–50 步统计。
- `TPS`：所有 rank 每步输入 token 总数除以时间。
- `HBM`：所有 rank 的框架分配峰值最大值。
- `Steady samples/s`：GBS 除以稳态 Step wall 均值。
- `线性度`：多机吞吐相对单机吞吐的比例，再除以理想节点倍数。
- `有效 MFU`：有效训练 FLOPs 除以“卡数 × 峰值 TFLOP/s × 步时”。
- `HFU`：有效 FLOPs 加 checkpoint 重算 FLOPs，不能作为有效 MFU 验收值。
- `CommNotOvl`：Insight 关键路径中未与计算重叠的通信时间。
- `HCCL total`：HCCL 算子耗时累计，可重叠，不能直接加到 Stage。
- `Bubble`：流水线气泡字段；为 0 不代表没有普通设备空档。

## 7. 完成与验收

每轮必须满足：

- 所有参与 rank 的退出码为 0。
- 50/50 步完成，loss 和 grad norm 为有限值。
- 每个 rank 的 Step wall、投影审计和 Profiler 文件齐全。
- 两机或四机 workload 签名一致。
- 全 rank HBM 未达到 96 GB 上限。

最终自动生成：

```text
artifacts/runs/mfu/scaling-scale1114-03/comparison.md
artifacts/runs/mfu/scaling-scale1114-03/comparison.csv
artifacts/runs/mfu/scaling-scale1114-03/summary.json
```

