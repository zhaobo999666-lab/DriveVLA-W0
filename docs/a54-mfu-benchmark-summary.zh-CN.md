# A54 单机 8 卡 MFU 性能实验总结

实验日期：2026-11-11。目标：用同一模型和真实数据建立可复算的单机性能基线，修复性能档，再找出本轮 MFU 最高的 batch。

**结果：** 每卡 batch 2 时，`safe` 与修复后的 `fast` 均跑完 50 步，稳态 useful MFU 分别为 **8.74%** 和 **34.32%**。开启融合 gate/up 后，**每卡 batch 27 的短测最高，为 47.76%**；50 步确认因 A54 被其他任务占用而未进入训练。日常长跑建议优先验证 batch 24：短测只低 0.20 个百分点，端到端吞吐接近，rank 0 峰值显存少约 5 GB。

## 1. 资源与配置

| 项目 | 本次使用 |
| --- | --- |
| 宿主机 | A54，`liteserver-a5-00004`，8 × Ascend 950DT |
| 容器、发布版本 | `zhaobo-drivevla-a54`；训练代码与共享 release 同步至 `d9efa30` |
| 模型 | Qwen3-VL-8B-Instruct；容器内 `/work/zhaobo/models/Qwen3-VL-8B-Instruct` |
| ROSS 权重 | Stable Diffusion v1.5 的 UNet 和 VAE；`/work/zhaobo/models/stable-diffusion-v1-5` |
| 动作 tokenizer | 仓库内 `configs/fast` |
| 数据 | OpenScene / NAVSIM trainval 子集，从 0–39 号相机分片准备；61,898 条训练样本，非全量 trainval |
| 样本索引、相机图像 | `/work/zhaobo/data/prepared-trainval/train.pkl`；`/work/zhaobo/data/raw-camera/openscene-v1.1/sensor_blobs/trainval` |
| 公共训练参数 | 单机 8 进程，BF16，梯度累积 1，seed 42，`max_pixels=903168`，`min_pixels=12544`，`model_max_length=4096`；每步记日志，关闭 profiling 和 checkpoint |
| 批量 | MBS 是每卡 batch；本次 GBS = 8 × MBS，例如 MBS 2 → GBS 16，MBS 27 → GBS 216 |

宿主机共享根目录是 `/mnt/sfs_turbo/zhaobo/drivevla-w0`，容器内共享资源从 `/work/zhaobo` 访问。数据准备流程见 [A5 执行手册](zhaobo-a55-a56-runbook.zh-CN.md)。

| 配置 | `safe` | 修复后的 `fast` |
| --- | --- | --- |
| 注意力 | SDPA | FlashAttention-2 |
| DeepSpeed | `zero3_offload.json`，参数和优化器状态卸载到 CPU | `zero2.json`，无 CPU 卸载 |
| DataLoader | 0 个后台 worker，关闭 DataLoader 锁页内存 | 每 rank 2 个 worker，开启 DataLoader 锁页内存 |
| 融合 gate/up | 关 | batch 2 对比关，batch 爬坡开 |

## 2. 从环境检查到性能测试

以下命令在 **A54 宿主机**执行。只在 8 张卡已分配给自己且进程表确认空闲时启动。

### 步骤 1：确认环境、发布版本和 NPU

```bash
SHARED=/mnt/sfs_turbo/zhaobo/drivevla-w0
RELEASE="$SHARED/artifacts/code/a55-a56"
BENCH="$SHARED/artifacts/code/mfu-benchmark-v2"

npu-smi info
ZB_NODE=a54 bash "$RELEASE/start-training.sh" status
docker exec zhaobo-drivevla-a54 df -h /dev/shm
ZB_MFU_PROFILE=fast bash "$BENCH/benchmark-mfu.sh" --dry-run
```

检查进程表、容器状态、`sync=ok` 和模型数据路径。空闲时每卡仍约有 4.8 GB HBM 占用，旧状态脚本会误报 `8/8`；以逐卡进程表为准。新启动器会再次核对进程和 HBM，并检查端口 29555（训练）和 29556（HCCL 验收）。首次实际启动会自动跑 HCCL，8 个 rank 的归约期望值为 **36**。

容器重建后如缺少 Transformers 或 tokenizers，按原部署方式补齐并验依赖：

```bash
docker exec zhaobo-drivevla-a54 bash --noprofile --norc -e -c '
  source /work/zhaobo/bootstrap/enter-env.sh
  env -u PYTHONPATH python -m pip install -e ./reference/transformers
  env -u PYTHONPATH python -m pip check
'
```

### 步骤 2：跑同 batch 的 safe / fast 基线

```bash
ZB_MFU_PROFILE=safe ZB_MBS=2 bash "$BENCH/benchmark-mfu.sh"
ZB_MFU_PROFILE=fast ZB_MBS=2 bash "$BENCH/benchmark-mfu.sh"
```

两条命令分别执行 HCCL 验收、32 条训练样本 CPU 形状采样、同配置 3 步冒烟和 50 步训练，并写出 `summary.txt`。这是当前推荐复跑命令：实际历史 `safe` 基线用 v1，修复后的 `fast` 基线用 v2，因前一轮已做过冒烟，实测时对该次复测设置了 `ZB_MFU_SMOKE=0`。两次正式基线均用第 6–50 步统计，未开启 gate/up。

脚本分工：

| 仓库脚本 | 作用 |
| --- | --- |
| [benchmark-mfu.sh](../scripts/a5/benchmark-mfu.sh) | 检查节点、空闲和资源；调用验收、训练和汇总；产物按 run ID 隔离 |
| [check-hccl.py](../scripts/a5/check-hccl.py) | 8 rank 单机归约验收 |
| [train_qwen_vla_navsim.sh](../scripts/scripts_train/train_qwen_vla_navsim.sh) | 实际传参给 torchrun 和训练入口 |
| [mfu-fast-bin/torchrun](../scripts/a5/mfu-fast-bin/torchrun)、[npu_zero2_compat.py](../utils/npu_zero2_compat.py) | 只在 benchmark 的 ZeRO-2 训练进程应用梯度检查兼容逻辑 |
| [benchmark_mfu.py](../utils/benchmark_mfu.py)、[flops_mindspeed.py](../utils/flops_mindspeed.py) | 采样真实训练张量、计算骨干 FLOPs、从 `trainer_state.json` 汇总 MFU |
| [sweep-mfu.sh](../scripts/a5/sweep-mfu.sh) | 逐档试 batch；遇到明确显存不足则停止爬坡；按短测最高 MFU 选档再跑 50 步 |

### 步骤 3：开启 gate/up，做 batch 爬坡

```bash
bash "$BENCH/sweep-mfu.sh" 8 16 24 27 32
```

每档短测跑 12 步、取第 6–12 步，不另外重复 3 步冒烟；该档失败就不会产出 MFU。首次实际跑到 batch 16 时遇到共享内存错误；扩容后用相同脚本从 16 继续：

```bash
bash "$BENCH/sweep-mfu.sh" 16 24 27 32
```

如果只重跑最高短测档的 50 步确认，不需要重跑整个扫描：

```bash
ZB_MFU_PROFILE=fast ZB_MBS=27 ZB_FUSED_GATEUP=1 \
ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10 bash "$BENCH/benchmark-mfu.sh"
```

日常验证将 `ZB_MBS` 改为 24。备用路线是 `ZB_FAST_ZERO_STAGE=3`（FA2 + 无 CPU 卸载的 ZeRO-3），本轮未用它产出性能结果。

## 3. 问题、定位和处理

| 问题 | 证据与处理 |
| --- | --- |
| 旧脚本把空闲卡判成忙；旧 HCCL 验收只认 16 rank | 逐卡无进程且利用率为 0，但基线 HBM 超过旧阈值。新性能启动器同时核对进程表与显存；共享发布版本已支持 8 rank。 |
| 重建容器后状态显示 READY，但缺 Python 包 | 容器内导入报 `No module named transformers`。按步骤 1 安装仓库自带 Transformers 后，`pip check` 通过，真实样本读取成功。 |
| 原始 fast 在第一步报 `aclnnAmax 561000` | 两次都在 rank 7 的 ZeRO-2 梯度溢出检查失败，报 2 字节目标与约 2.97 GB 源存储不匹配。独立大张量单卡归约正常，单纯按大逻辑张量分块也未修好整训。训练日志最终确认触发对象是引用大存储的 **单元素 BF16 视图**。v2 对该视图直接做 `isfinite`，对超大逻辑张量保留分块检查；有限值、NaN、正负 Inf 已验证。未关闭溢出检查，也未修改容器安装的 DeepSpeed。 |
| batch 16 的 DataLoader 报 `unable to allocate shared memory(shm)` | 当时个人容器的私有 `/dev/shm` 是 16 GiB。核对容器挂载与私有 IPC 后，临时将该容器的 tmpfs 上限调到 96 GiB；宿主机 `/dev/shm` 未修改。之后 batch 16–32 均完成短测。重建容器时用 `--shm-size=96g` 保留设置；不要把这个报错当成 NPU 显存 OOM。 |
| batch 27 的 50 步确认在初始化时失败 | `HcclBroadcast` 报 `EI0020`，NPU 端口 `16666` 已绑定；复查时 8 张卡上都有其他进程，自己的训练容器无训练进程。未终止其他任务，长跑需等独占资源后重跑。 |

扩容时的宿主机命令如下。只用于已核对属于自己的运行中容器，需要宿主机管理员权限；正常启动新容器时优先设定 `--shm-size`。

```bash
box=zhaobo-drivevla-a54
docker inspect -f '{{.Name}} {{.HostConfig.IpcMode}}' "$box"
docker inspect -f '{{range .Mounts}}{{println .Source .Destination}}{{end}}' "$box"
pid=$(docker inspect -f '{{.State.Pid}}' "$box")
findmnt -N "$pid" --mountpoint /dev/shm -n -o SOURCE,FSTYPE,OPTIONS
# 确认 IPC 为 private、/work/zhaobo 指向自己的工作区、/dev/shm 是容器的 tmpfs 后执行：
nsenter -t "$pid" -m -r -- mount -o remount,size=96g /dev/shm
docker exec "$box" df -h /dev/shm
df -h /dev/shm
```

## 4. 数据口径与实测结果

从真实 NAVSIM 训练数据加载器采 32 条样本：有效 LLM 序列长度中位数 **1767**，每样本前一帧、当前帧各 **3432** 个 ViT patch（中位数）。分别计算每条样本的 FLOPs 再平均，得到 **1.173843×10¹⁴ FLOPs/样本**。避免用中位长度直接代入注意力的平方项。

```text
useful MFU = 平均每样本骨干 FLOPs × GBS / (NPU 数 × 稳态平均步时 × 每卡峰值)
```

这里用 `step_time_full_s`，包含优化器更新，不含等待 DataLoader；端到端吞吐用全程样本数除以 Trainer 的 `train_runtime`，包含热身。峰值 **486 TFLOP/s/卡** 是假设值；分子计 Qwen3 文本和 ViT，不计 ROSS 和重计算，但实际训练跑了 ROSS。参考 [npu-port.md](../npu-port.md) 的 48.4% 含 ROSS，数据及统计窗口也有差异。

| 档位 | MBS | 步数 | 稳态均值 (秒) | Useful MFU | 端到端样本/秒 | Rank 0 峰值 HBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| safe | 2 | 50 | 5.527 | 8.74% | 2.28 | 19.4 GB |
| fast，修复版 | 2 | 50 | 1.408 | 34.32% | 10.25 | 46.1 GB |
| fast + gate/up | 8 | 12 | 4.284 | 45.10% | 12.29 | 47.6 GB |
| fast + gate/up | 16 | 12 | 8.223 | 47.00% | 13.20 | 61.0 GB |
| fast + gate/up | 24 | 12 | 12.188 | 47.56% | 13.51 | 74.4 GB |
| fast + gate/up | 27 | 12 | 13.654 | **47.76%** | 13.54 | 79.4 GB |
| fast + gate/up | 32 | 12 | 16.679 | 46.34% | 13.34 | 87.7 GB |

## 5. 原始产物索引

以下均相对于共享盘 `/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/`：

| 产物 | 路径 |
| --- | --- |
| safe 50 步 | `mfu-a54-20261111-113939-2168880/summary.txt` |
| 原始 fast 的失败日志 | `mfu-a54-20261111-115338-2200665/smoke-fast.log` |
| fast 修复后 batch 2 的 50 步 | `mfu-a54-20261111-120635-2234293/summary.txt` |
| batch 8 结果、首次 batch 16 失败 | `sweep-liteserver-a5-00004-jKYflp/results.tsv`、`sweep-liteserver-a5-00004-jKYflp/b16.log` |
| 扩容后 batch 16–32 的扫档表 | `sweep-liteserver-a5-00004-lfPkm6/results.tsv` |
| batch 27 短测最高点 | `mfu-a54-20261111-122947-2297345/summary.txt` |
| batch 27 长跑初始化冲突 | `mfu-a54-20261111-124057-2325396/train-fast.log` |

每个成功的运行目录还包含 `run-info.txt`、`shapes.json`、训练日志和 `train-<profile>/navsim/trainer_state.json`，可据此重算结果。早先 3 步冒烟估算的 4.87% 不用于正式比较。
