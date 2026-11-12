# A5 上运行 Qwen3-VL + ROSS 性能基线

A55/A56 已释放。当前先用已准备好的 **A52、A53 或 A54 任意一台**建立单机 8 卡性能基线。A51 要先准备自己的容器和代码环境。以下保留原双机流程供后续扩展。

在已分配给你且 8 卡空闲的 A52/A53/A54 **宿主机终端**执行：

```bash
bash /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/mfu-benchmark-v2/benchmark-mfu.sh
```

命令使用 Qwen3-VL-8B + ROSS、已准备的 trainval 子集、每卡 batch 2（GBS=16），先跑单机 HCCL 和真实数据形状采样，再对 `safe`、`fast` 依次做 3 步冒烟和 50 步测量，每完成一档就保存报告。性能跑关闭 profiling 和模型保存。占用检查未通过或任一验收失败即停止。只跑稳定档可在命令前加 `ZB_MFU_PROFILE=safe`；只验证路径与配置可加 `--dry-run`。后续需要试更大的 batch，可在命令前加 `ZB_MBS=4`，每档都会重新冒烟。

结果写到 `/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/runs/mfu/<run-id>/summary.txt`。同一份报告给出稳定步时、估算 useful MFU、端到端 samples/s 和 rank 0 峰值 HBM。MFU 的 FLOPs 用真实训练数据加载器采样，按样本平均，分子仅算 Qwen3 文本和 ViT，峰值按假设的每卡 486 TFLOP/s。端到端数据取 Trainer 50 步总时长（含热身）；稳定步时含优化器，不含等数据的时间。优化归因时，同一台机器、同一批数据和 batch，每轮只改一个开关。

### A54 的 ZeRO-2 兼容修复和 batch 爬坡

在 A54，原版 `fast` 的 HCCL 正常，但 rank 7 在第一步梯度溢出检查中报 `aclnnAmax 561000`。训练日志验证了触发处是一个只有一个 BF16 元素、背后引用约 2.97 GB 梯度存储的视图。`mfu-benchmark-v2` 的 ZeRO-2 工作进程对这种视图直接做 `isfinite`；超大逻辑张量则分块检查。NaN/Inf 检测保留，容器安装的 DeepSpeed 不变。若要走备用方案，可用 `ZB_MFU_PROFILE=fast ZB_FAST_ZERO_STAGE=3`，保留 FA2 但使用无 CPU 卸载的 ZeRO-3；与 ZeRO-2 结果分开记录。

稳定通过每卡 batch 2 后，可以用 `mfu-benchmark-v2/sweep-mfu.sh` 跑 8、16、24、27、32 的短测（也可用位置参数指定 batch 列表）。爬坡开启融合 gate/up；遇到确认的显存不足停止，取短测 MFU 最高的 batch 再跑 50 步确认。短测、失败日志及确认结果分别保存。

A54 的原容器只有 16 GiB 私有 `/dev/shm`，batch 16 的 DataLoader 预取在这里报共享内存分配失败。已临时将该容器的私有 `/dev/shm` 调到 96 GiB。重建容器后应按实际大 batch 需求配置 `--shm-size` 并在容器内用 `df -h /dev/shm` 检查；不要把共享内存报错记作 NPU 显存不足。

2026-11-11 A54 实测：修复后的 ZeRO-2、每卡 batch 2 完成 50 步，useful MFU **34.32%**（同 batch 的 `safe` 为 **8.74%**）。开启融合 gate/up 后，12 步短测（取第 6–12 步）中每卡 batch 8 / 16 / 24 / 27 / 32 分别为 **45.10% / 47.00% / 47.56% / 47.76% / 46.34%**。batch 27 短测最高，50 步确认因另一任务占用 A54、HCCL 端口绑定冲突而未开始。batch 24 峰值显存比 27 少约 5 GB、端到端吞吐近似，适合优先长跑。

## 历史 A55 + A56 双机流程

两台机器共享 `/mnt/sfs_turbo/zhaobo/drivevla-w0`，不需要重新下载权重或数据。训练脚本用静态 torchrun 会合，每台 8 个进程，共 16 个进程。

| 节点 | 节点编号 | HCCL Host IP | 容器 |
| --- | --- | --- | --- |
| A55 | 0 | 10.0.16.77 | `zhaobo-drivevla-a55` |
| A56 | 1 | 10.0.16.230 | `zhaobo-drivevla-a56` |

两台都使用 `enp39s0f3`，会合地址为 A55 的 `10.0.16.77`。训练端口为 29555，通信验收端口为 29556。已测两台宿主机能互相 ping，尚未用 HCCL 验证设备间链路。

启动器（`start-training.sh` / `a55-a56.sh`）支持三个可选参数：**档位**（`safe`|`fast`）、**每卡 batch**、**模型**（`qwen25`|`qwen3vl`）：

| 模型参数 | 权重 | max_pixels | 说明 |
| --- | --- | --- | --- |
| `qwen25`（默认） | Qwen2.5-VL-7B-Instruct | 150528 | real50-01~04 验证过的基线 |
| `qwen3vl` | Qwen3-VL-8B-Instruct | 903168 | **npu-port 最佳点（L≈1774），与 `utils/flops_mindspeed.py` 口径完全一致，是性能优化任务的对照主路线** |

代码部署在 `qwen3vl-backbone` 分支（当前生效的提交以共享盘 `artifacts/code/a55-a56/REVISION` 为准，`bash a55-a56.sh status` 会打印 `release=<rev> repo=<node>@<rev>`），vendored transformers 含 `qwen3_vl` 与 `modeling_qwen3_vl_ross.py`。启动器默认已启用 npu-port 附录中保留的全部加速特性：`CPU_AFFINITY_CONF=1`、`expandable_segments:True`、`ROSS_STEP_TIMING=1`、valid-token CE、融合 RMSNorm+RoPE。

## 代码同步：仓库是真相源，release 是执行副本（2026-11-10 新增）

同一份代码在三处存在，职责必须分清，否则会出现"改了代码但训练跑的是旧版"这种无声故障：

| 位置 | 角色 |
| --- | --- |
| 本地工作区 `D:\OTT9\DriveVLA-W0` | 编辑与提交，`qwen3vl-backbone` 分支 |
| 各机仓库 `/data1/zhaobo/drivevla-w0-<node>/code/DriveVLA-W0` | 只读镜像，必须与发布提交一致 |
| 共享盘 `/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56` | **真正被执行的那份**（启动器强制 `HERE == RELEASE`，也只能从这里跑） |

改完代码的标准流程（四步）：

```bash
# 1) 本机提交后打成增量 bundle（节点不必连 GitHub）
git bundle create artifacts/code/a55-a56/qwen3vl-backbone-<rev>.bundle qwen3vl-backbone --not <上一个已发布 rev>
scp artifacts/code/a55-a56/qwen3vl-backbone-<rev>.bundle ott9-a52:/tmp/
ssh ott9-a52 'cp -f /tmp/qwen3vl-backbone-<rev>.bundle /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56/'

# 2) 每台机器把仓库快进到该提交（分支已被 checkout 时不能直接 fetch 进同名分支）
cd /data1/zhaobo/drivevla-w0-<node>/code/DriveVLA-W0
git fetch /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56/qwen3vl-backbone-<rev>.bundle qwen3vl-backbone
git merge --ff-only FETCH_HEAD

# 3) 任一节点把仓库发布到 release（只需一台；内容相同的重复发布是幂等的）
cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56
bash a55-a56.sh publish
# Publishing /data1/.../code/DriveVLA-W0@a1b2c3d -> /mnt/sfs_turbo/.../artifacts/code/a55-a56
#   a55-a56.sh                         0755  <md5>
#   ...
# REVISION -> a1b2c3d...
# publish OK (6 files)

# 4) 体检一眼确认是否同步
bash a55-a56.sh status
# node=a52 ... container=Up
# release=a1b2c3d repo=a52@a1b2c3d  sync=ok
```

`publish` 做的事：把 6 个受管文件（`a55-a56.sh`、`start-training.sh`、`check-hccl.py`、`train_qwen_vla_navsim.sh`、`prepare-a55.sh`、本 runbook）从本机仓库 `install` 到 release，逐个打印 md5，写 `REVISION`，缺 `DriveVLA-W0-<rev>.bundle` 时补一份完整包（供 `prepare-a55.sh` 冷启动克隆），最后再整体比对一次。替换用 `tmp+mv` 原子完成——直接覆盖会让正在执行的 bash 按字节偏移续读到错位内容。

兜底校验：`check` / `smoke` / `train` 启动时都会比对 release 与仓库的 md5，不一致就打印

```
WARN: release/train_qwen_vla_navsim.sh differs from repo/scripts/scripts_train/train_qwen_vla_navsim.sh
WARN: 1/6 release file(s) out of sync with /data1/.../DriveVLA-W0@a1b2c3d.
WARN: the run uses the RELEASE copies, not the repo ones.
WARN: refresh with: bash /mnt/.../a55-a56.sh publish
```

以及

```
ERROR: Repo /data1/... is at 9f8e7d6 but the release was published from a1b2c3d; run publish or check out the published revision
```

默认只告警不拦（避免机器被别人临时占用时卡住），需要强一致时加 `ZB_STRICT_RELEASE=1`，把告警升级为失败。`prepare-a55.sh` 的期望提交也改成读同一个 `REVISION` 文件，不再各处硬编码。


## 数据集：mini 学习子集 → trainval（2026-11-10 新增）

默认 `data/prepared` 是 **OpenScene mini 的单个 log**（797 训练样本 / 16 验证样本），只够跑通链路与测步时。要真实规模的数据就换 trainval：

| 包 | 体积 |
| --- | --- |
| `openscene_metadata_trainval.tgz` | 7.05 GB（全部 log 的 pkl） |
| `openscene_sensor_trainval_camera_*.tgz` | 200 片共 **1242 GB**，单片 4.6–19.3 GB |

实测 hf-mirror 带宽：单流 ≈3 MB/s，8 并发 ≈11 MB/s，16 并发 ≈29 MB/s。**40 片（≈250 GB）约 2–3 小时**，全量 200 片约 8–12 小时。

三步走（都在任一节点上做，路径是共享盘，其他节点立即可见）：

```bash
cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56
# 1) 元数据（约 10 分钟）
docker exec zhaobo-drivevla-a52 env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  hf download OpenDriveLab/OpenScene --repo-type dataset --include "openscene-v1.1/openscene_metadata_trainval.tgz" \
  --local-dir /work/zhaobo/downloads/openscene --max-workers 8

# 2) 相机分片：下载→解压→删 tgz，逐片落 marker，可断点续跑（后台跑，不受 ssh 断开影响）
docker exec -d zhaobo-drivevla-a52 bash /work/zhaobo/artifacts/code/a55-a56/fetch-openscene-trainval.sh 0 39 12
tail -f /data1/zhaobo/drivevla-w0-a52/manifests/fetch-openscene-trainval.log

# 3) 预处理（同一个 dataset 模式，靠环境变量切分片与规模）
#    非 a55 节点（a52/a53…）跑 dataset 需显式给节点身份：prepare-a55.sh 自 2026-11-10 起
#    允许 dataset/models 在 peer 节点执行（纯 CPU，不碰 NPU），environment 仍只能在 a55 做
ZB_NODE=a52 ZB_EXPECTED_HOST=liteserver-a5-00002.novalocal \
ZB_SPLIT=trainval ZB_SHARDS="$(seq -s ' ' 0 39)" \
ZB_MAX_TRAIN=200000 ZB_MAX_VAL=2000 ZB_PER_LOG_CAP=400 \
  setsid nohup bash prepare-a55.sh dataset > /data1/zhaobo/drivevla-w0-a52/manifests/prepare-dataset.console.log 2>&1 < /dev/null &
# 进度看 status / 日志（状态机 RUNNING <stage> → READY / FAILED exit=N）
cat /data1/zhaobo/drivevla-w0-a52/manifests/prepare-dataset.status
tail -f /data1/zhaobo/drivevla-w0-a52/manifests/prepare-dataset.console.log
```

产物落 `data/prepared-trainval/`（**不会覆盖** mini 的 `data/prepared`），`data_manifest.json` 里记录实际用到的 log 列表与样本数。

`ZB_PER_LOG_CAP` 很关键：`support.py` 会逐帧解码图像做校验，全量 log 不设上限时既慢又吃内存。它是"每个 log 最多取多少样本"，0 表示不限（默认，保持 mini 行为不变）。

**实测（2026-11-10 23:31–23:39，40 片 / 244 GB）**：212 个训练 log + 54 个验证 log，产出 **61 898 训练 / 2 000 验证**样本，pkl 共 64 MB，全程约 8 分钟。注意 `ZB_PER_LOG_CAP=400` 是实际瓶颈（212 log × 400 ≈ 85k 上限），所以 `ZB_MAX_TRAIN=200000` 没被吃满；要更多样本就调大 cap 或补更多分片。

**坑（已修）**：`raw-logs/openscene-v1.1/meta_datas/` 下同时存在 `mini/` 和 `trainval/`，而 `support.py` 是 `rglob("*.pkl")` 递归扫描，会把同一个 log 扫两次并报 `ValueError: Duplicate raw log name`。现在 `prepare-a55.sh` 按 split 精确定位到 `meta_datas/<split>/`，找不到才退回 `raw-logs` 根。

训练侧切到新数据：启动器已支持成对覆盖（**必须两个一起给**，pkl 里存的是相对路径，`data_root` 不匹配就会找不到图）：

```bash
export ZB_DATA=/work/zhaobo/data/prepared-trainval/train.pkl
export ZB_DATA_ROOT=/work/zhaobo/data/raw-camera/openscene-v1.1/sensor_blobs/trainval
bash a55-a56.sh smoke <run-id> fast 2 qwen3vl 1   # 先预演：ZB_DRYRUN=1 会打印实际数据路径
```

两者缺一即报错（`ZB_DATA` 没有配对的 `ZB_DATA_ROOT` 时启动器直接 die）。注意 `enter-env.sh` 里的 `RGB_ROOT` 仍指向 `sensor_blobs/mini`，不覆盖 `ZB_DATA_ROOT` 就会拿 trainval 的 pkl 去 mini 目录找图。


## 机器池与配对：A52 / A53 / A54 任意两台（2026-11-10 起）

A55/A56 已释放。当前可用的 A5 机器是一个**共享池**（其他同学也在排队用），因此启动器按"任意两台配对"设计，机器名不写死。

| 节点 | 主机名 | `enp39s0f3` IP | 初始化状态（11-10 18:50） |
| --- | --- | --- | --- |
| A52 | liteserver-a5-00002 | 10.0.16.75 | ✅ READY，容器 `zhaobo-drivevla-a52` |
| A53 | liteserver-a5-00003 | 10.0.16.231 | ✅ READY，容器 `zhaobo-drivevla-a53` |
| A54 | liteserver-a5-00004 | 10.0.16.118 | ✅ READY，容器 `zhaobo-drivevla-a54` |

三台两两互 ping 全通；三台环境已全部初始化（镜像 load、5 项挂载、torch_npu/deepspeed/diffusers、`zero2.json` 与 `zero3_offload.json` 均在），代码仓库已随 `publish` 快进到 `REVISION` 记录的提交（见上一节）。

**只支持 1 台或 2 台**（`nodes=1|2`），不支持三机同跑——本项目的并行策略按 8 卡/节点 × 1~2 节点设计。

### 1. 开跑前先体检，挑出真正空着的两台

```bash
cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56
bash start-training.sh status
# node=a52 host=liteserver-a5-00002 hccl=10.0.16.75 env=READY npu_busy=8/8 container=Up
```

`npu_busy` 不是 0 就别用这台——11-05 与 11-10 都出现过 vLLM 纯驻留占满 8 卡（每卡 59 GB/98 GB）。启动器里也有强检，会在真正启动前拦住，但先体检能省掉一轮沟通。

### 2. 配对：两台给同一个 `ZB_PAIR`，不用手抄 IP

```bash
# 两台都保持同一个 ZB_PAIR；只有 rank 和节点名不同
# A52（本机 = rank 0，会把 10.0.16.75 写到共享盘）
export ZB_PAIR=zb5253 ZB_NODE=a52 ZB_NODE_RANK=0
# A53
export ZB_PAIR=zb5253 ZB_NODE=a53 ZB_NODE_RANK=1
```

rank0 启动时把本机 `enp39s0f3` IP 写入 `artifacts/runs/pairs/<ZB_PAIR>/master-<run-id>.ip`，rank1 轮询该文件（最长 5 分钟）自动取到会合地址。**先起 rank0，再起 rank1**；`ZB_MASTER_ADDR` 仍可显式覆盖。

其它组合同理，只是换名字：A52+A54、A53+A54 都可以，rank0 给谁不影响性能（会合流量很小）。

### 3. 预演（不碰 NPU、不占端口）

调参数、换机器、改 runbook 后先跑一次，确认节点/rank/会合地址/输出目录都对：

```bash
ZB_DRYRUN=1 bash start-training.sh check dryrun01 fast 2 qwen3vl
# [pair] a52 rank0 advertises 10.0.16.75 for run dryrun01
# [dry-run] node=a52 rank=0 hccl=10.0.16.75 master=10.0.16.75 port=29556 nodes=2
```

### 4. 正式流程（两台执行同样的命令，只是环境变量不同）

```bash
bash start-training.sh check comm-01 fast 2 qwen3vl   # 16 rank HCCL
bash start-training.sh smoke smoke01 fast 2 qwen3vl   # 3 步，存 SMOKE OK 标记
bash start-training.sh train real50-q3 fast 2 qwen3vl # 50 步
```

`fast` 档的 `train` 会检查本机日志里是否已有同档 `HCCL OK` / `SMOKE OK`，缺一个就拒绝启动——这是防止"跳过通信验收直接长跑"的硬门禁。

### 5. 共享盘的注意事项

- 运行 ID 全局唯一：`artifacts/runs/a55-a56/<mode>-<run-id>` 已存在且非空时会被拒绝，避免撞掉别人（或自己上一轮）的产物。
- 日志落在各节点本地 `/data1/zhaobo/drivevla-w0-<node>/runs/a55-a56/`，文件名含 mode/run/profile/model/batch/nodes，多组并行时不会混。
- 权重与数据都只有一份（共享盘 `models/`、`data/`），换机器不需要重新下载；Qwen3-VL-8B-Instruct（17 GB）已在位。

## 迁移：A55/A56 已释放，改用 A52 + A53（2026-11-10）

启动器已泛化为任意 a5 节点对：主机名能识别 a55/a56（历史），其他节点用环境变量声明身份，IP 全部从 `enp39s0f3` 自动探测，不再硬编码。

**资产盘点**（迁移不丢东西的关键在 SFS 共享盘）：

| 资产 | 位置 | 状态 |
| --- | --- | --- |
| 已验收镜像导出（tar + SHA256 + image.id） | 共享盘 `artifacts/images/a55-qwen-v1`（17 GB） | ✅ 直接 load 到新节点 |
| 代码 bundle + 启动器 + runbook | 共享盘 `artifacts/code/a55-a56` | ✅ |
| 训练数据 / 权重（Qwen2.5-VL、SD1.5、Qwen3-VL-8B） | 共享盘 `data` / `models` | ✅ |
| a55/a56 工作区备份 | 共享盘 `machine-release-backup-20261110/` | ✅ |
| a55/a56 容器与本地盘工作区 | a55/a56 `/data1` | ❌ 随机器释放 |

**初始化（已完成，换新机器时才需要重跑）**：

```bash
# 任意节点：校验 SHA256 → docker load → 建容器/工作区 → 依赖与训练入口验收
ZB_NODE=a53 ZB_NODE_RANK=0 bash a55-a56.sh prepare-peer
```

`prepare-peer` 不会合，无需会合地址；它只要求共享盘有可用的镜像导出。

## 快速启动

在两台宿主机进入 `/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56`（机器池与配对方式见上一节）。

**登录到机器后的最短路径**（节点名脚本按 `/data1` 目录自动识别，只需给 rank）：

```bash
cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56

# —— 第 1 台：rank 0 ——
export ZB_PAIR=zb01 ZB_NODE_RANK=0
bash start-training.sh status                              # 确认 env=READY 且 npu_busy=0/8
bash start-training.sh check  comm-01  fast 2 qwen3vl
bash start-training.sh smoke  smoke01  fast 2 qwen3vl
bash start-training.sh train  real50-q3 fast 2 qwen3vl

# —— 第 2 台：rank 1（同一个 ZB_PAIR，其余命令完全相同）——
export ZB_PAIR=zb01 ZB_NODE_RANK=1
bash start-training.sh status
bash start-training.sh check  comm-01  fast 2 qwen3vl
bash start-training.sh smoke  smoke01  fast 2 qwen3vl
bash start-training.sh train  real50-q3 fast 2 qwen3vl
```

要点：

- **先起 rank0，再起 rank1**，两台相差控制在 1 分钟内（check 的 rendezvous 超时是 120 s，train 会合超时更长，但拖久了也会失败）。
- `export` 只对当前 shell 有效；重新登录要重新 export，或写成 `ZB_PAIR=zb01 ZB_NODE_RANK=0 bash start-training.sh ...` 的行内形式。
- 单机自测不需要 `ZB_PAIR`：`bash start-training.sh train real50-s1 fast 24 qwen3vl 1`。
- 每一步的 `check`/`smoke` 会在本机日志留 `HCCL OK`/`SMOKE OK` 标记，`fast` 档的 `train` 缺标记会拒绝启动。

**逐级流程**（同上，只是展开说明）：

1. 两台各跑 `bash start-training.sh status`，确认 `env=READY` 且 `npu_busy=0/8`。
2. 两台 `export ZB_PAIR=<同一个配对ID>`，加上本机 `ZB_NODE_RANK=<0|1>`，再用 `ZB_DRYRUN=1` 跑一次 `check` 预演。
3. 两台各执行 `bash start-training.sh check comm-01 fast 2 qwen3vl`，通过后各执行 `smoke`，最后各执行 `train real50-q3 fast 2 qwen3vl`。
4. 单机：追加最后一个参数 `1`（如 `train real50-s1 fast 24 qwen3vl 1`），不需要 ZB_PAIR/RANK。

启动脚本不写死主机名：rank 由 `ZB_NODE_RANK` 指定，节点名按 `/data1` 反推，rank0 的会合 IP 自动写入共享盘供 rank1 读取；训练默认 50 步、每卡 batch 为 2、每步记录日志、关闭 profiling。为避免自动续训，已有产物或最终训练状态的运行 ID 会被拒绝；需要明确续训时才使用 `bash a55-a56.sh train <原运行ID> ...`。

## 性能档位（对照 npu-port.zh-CN.md 附录，2026-11-10 增补）

real50-04（2026-11-10）用 Qwen2.5-VL 的 `safe` 配置验证了 50 步双机训练和 checkpoint 存取。第 6–50 步 `step_time_full_s` 中位数 4.466 s，均值 4.646 s。早先的 3.08% 来自 Qwen3 解析器误读 Qwen2.5 配置，不能作为 MFU；两次 ROSS 计数也未完成。建立优化基线请用本文顶部的 Qwen3 单机命令。历史配置对照：

| npu-port 附录最佳点 | real50-04 实际 | 处置 |
| --- | --- | --- |
| `--attn_type fa2`（Qwen2.5 上 +83%） | sdpa | fast 档已切 fa2 |
| ZeRO-2（无每步参数 all-gather） | ZeRO-3 + optimizer/param 双 offload CPU | fast 档已切 zero2.json |
| pinned loader + memlock | pin false + workers 0 | fast 档已开 pin true + workers 2（容器 memlock 受限时回退 false） |
| 融合 RMSNorm+RoPE、valid CE、expandable segments、CPU 亲和性 | 相同（脚本默认已启用） | 不变 |
| 融合 gate/up | Qwen3 代码支持 `ROSS_FUSED_GATEUP` | 初始基线关闭，后续可单独对照 |
| 单卡 bs 24–27（8B ROSS） | bs 2（7B+SD1.5） | fast 档冒烟通过后逐步爬坡 |

启动器支持第三个参数选档、第四个参数调每卡 batch：

```bash
# 稳定档（默认，与 real50-04 相同）
bash start-training.sh check comm02
bash start-training.sh smoke smoke02
bash start-training.sh train real50-05

# 快速档（对照 npu-port 最佳点）；train 会被强制要求先跑通同档 check 和 smoke
bash start-training.sh check comm02 fast
bash start-training.sh smoke smoke02 fast
bash start-training.sh train real50-05 fast

# fast 冒烟通过后，batch 爬坡（每档都先 smoke 再 train）
bash start-training.sh smoke smoke03 fast 4
bash start-training.sh train real50-06 fast 4
```

爬坡与判据：

1. 先跑 `fast` 档 bs 2 的 50 步（real50-05），与 real50-04 对比步时（期望 fa2 + ZeRO-2 明显低于 4.35s）与 loss 曲线（两条曲线应基本重合，验证 fa2 数值无回归）。
2. bs 爬坡阶梯 4 → 8 → 12；出现 OOM 或步时方差变大就停在上一档。npu-port 的经验：ZeRO-2 下加大 batch 是免费的收益，本项目的爬坡上限需实测。
3. `pin_memory true` 若触发容器 pinned-host-memory 报错（脚本帮助注明 false 就是为绕开它），说明 `ulimit -l unlimited` 在容器内未生效——改回 safe 档或让管理员调容器 memlock，不要硬试。
4. Qwen3 已有 `ROSS_FUSED_GATEUP` 和 `ROSS_FUSED_QKV` 开关，初始基线均关闭。若单独测试，确认变量传到了容器内，并固定其余配置。

## 单机模式（8 卡，nodes=1）

`npu-port.zh-CN.md` 附录的全部实测数字（48.4% MindSpeed MFU、27.8k tokens/s、bs 24–27、步时 12.4–13.8s）都来自**单机 8 卡**。启动器支持第 6 参数 `nodes=1`（或环境变量 `ZB_NODES=1`），在本机单机运行，无需另一台机器：

```bash
# 单机逐级验收（fa2 + ZeRO-2 = npu-port 最佳点口径）
bash start-training.sh check  comm-s1  fast 2  qwen3vl 1
bash start-training.sh smoke  smoke-s1 fast 2  qwen3vl 1
bash start-training.sh train  real50-s1 fast 2  qwen3vl 1   # 普通训练会保存；测性能用本文顶部命令
```

单机模式的行为差异：

- **只在本机运行**：`--nnodes 1`、rank 固定 0、会合地址取本机 `enp39s0f3` IP，忽略 `ZB_NODE_RANK` / `ZB_MASTER_ADDR`；任何 A5 节点（含未映射的新机器）都能直接跑。
- run ID 查重与 fast 档 check/smoke 门禁都只看本机日志。
- 训练输出与双机共用 `artifacts/runs/a55-a56/train-<id>/`（共享盘），日志文件名带 `n1` 后缀。

单机可作为无跨机通信的参考。排障时可先单机复现，再上双机定位跨机链路。npu-port 的 48.4% 对应特定数据与计时口径，自己的基线以 `summary.txt` 为准。单机每卡 batch 2 时全局 batch 为 16；切到双机并保持每卡 batch 不变时全局 batch 为 32。

## Qwen3-VL-8B 双机训练（MFU 对照主路线）

目标：在双机 16 卡上复现 npu-port 附录的**配置**（`--max_pixels 903168`、fa2、ZeRO-2、valid CE、融合 RMSNorm+RoPE），用 `utils/flops_mindspeed.py` 按**本数据集实测 seqlens** 产出同口径 MFU，并与我们自己的 safe 档基线对比。

### 1. 准备权重（一次性）

Qwen3-VL-8B-Instruct 下载到共享盘（两机共用，容器内路径 `/work/zhaobo/models/Qwen3-VL-8B-Instruct`）：

```bash
# 在任一能访问共享盘的宿主机执行（当前为 A51/A52，约 17 GB）；国内走镜像：
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]" -i https://pypi.tuna.tsinghua.edu.cn/simple
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /mnt/sfs_turbo/zhaobo/drivevla-w0/models/Qwen3-VL-8B-Instruct
```

SD1.5（ROSS 头）已在 `/work/zhaobo/models/stable-diffusion-v1-5`，无需重下。启动器会在启动前检查该权重目录存在，缺失直接报错。

### 2. 逐级启动（每档先 check、smoke，再 train；两机同命令）

```bash
# 冒烟链路（fast 档 = fa2 + ZeRO-2 + pinned loader）
bash start-training.sh check  comm03 fast  2  qwen3vl
bash start-training.sh smoke  smoke03 fast  2  qwen3vl

# 双机起步：每卡 bs 2（全局 batch 32），50 步
bash start-training.sh train  real50-q3 fast 2 qwen3vl
```

启动前若尚未下载权重会直接被拦；`fast` 档的 train 会强制要求先通过同档 check + smoke。

### 3. 与 npu-port 最佳点的对应关系

| npu-port 最佳点配置 | 双机启动器落点 |
| --- | --- |
| `--max_pixels 903168` | `qwen3vl` 档自动带上。实际序列长度受原图尺寸及训练模板影响，以本次 `shapes.json` 为准 |
| `--attn_type fa2` | `fast` 档自动带上 |
| ZeRO-2（`zero2.json`，无 offload） | `fast` 档自动带上 |
| valid CE、融合 RMSNorm+RoPE、expandable segments、CPU 亲和性 | 启动器默认（与 npu-port 保留清单一致） |
| 参考文档的每卡 bs 24–27 | 本项目从每卡 2 起步，逐档冒烟并记录峰值 HBM |
| ROSS SD1.5 头 | 固定 `--sd_model_path` |

固定每卡 batch 比较 8 卡和 16 卡时，全局 batch 会翻倍，应用吞吐和扩展效率衡量；不要要求 loss 曲线逐步重合。换 batch、节点数或数据集后，先以目标配置冒烟，再跑正式测量。原启动器的 fast 门禁只检查历史日志中的档位，不保证目标 batch 已通过。

## 如何读 MFU 报告

- `summary.txt` 用稳定步第 6–50 步的 `step_time_full_s` **均值**换算 useful MFU，同时给出中位数以观察典型步时。训练器在优化器更新前打点，原始 JSON 的步号从 0 开始。
- `shapes.json` 记录实际训练数据加载器采样的 32 条样本：LLM 用有效 token 数，ViT 对每张图单独算 T×H×W。先对每条样本算 FLOPs，再平均，避免用单个中位序列长度代入注意力的平方项。换数据或图像尺寸必须重测。
- 端到端吞吐按全局样本数除以 Trainer 的训练总时长，包含热身和等数据的耗时。初始 A/B 固定节点、数据和 batch；`safe` 到 `fast` 是多项配置的整体收益。
- `npu-port.zh-CN.md` 的 48.4% 来自另一个数据形状和计时口径，只作参考。本文报告的 useful MFU 只计 Qwen3 文本和 ViT，未计 ROSS 和重计算；486 TFLOP/s 是每卡假设峰值。带 ROSS 的结果需单独列，不能与骨干口径混用。

## 1. 复用 A55 已验收的镜像

先在 A55 执行：

```bash
cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56
bash a55-a56.sh export-image
```

镜像会导出到共享盘的 `artifacts/images/a55-qwen-v1`，带镜像 ID 和 SHA256 清单，需预留约 17 GB。导出只读 A55 现有镜像，不会重建 A55 容器。

导出成功后在 A56 执行：

```bash
cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56
bash a55-a56.sh prepare-peer
```

此步骤校验并加载同一镜像，创建 A56 自己的容器和本地代码工作区，从共享 bundle 检出同一个提交，并执行依赖与训练入口的导入验收。

## 2. 两端分别启动，逐级验收

确认 A55 和 A56 各自的 8 张 NPU 都已分配给本次任务。在两台宿主机各开一个终端或 tmux，分别进入共享脚本目录。每个阶段都在两台执行同一条命令、使用同一个运行 ID；可先启动 A55，随即启动 A56。脚本按主机名设置 rank。

```bash
cd /mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/a55-a56
bash a55-a56.sh check comm01
```

只有两边各 8 个进程都输出 `HCCL OK`、归约结果为 136，才继续。再在两边执行：

```bash
bash a55-a56.sh smoke smoke01
```

3 步测试使用真实数据，不写训练权重。通过后，两边都执行：

```bash
bash a55-a56.sh train real50-01
```

默认每卡 batch 为 2、梯度累积为 1，所以双机全局 batch 为 32，比原单机的 16 大一倍。正式命令跑 50 步，并在第 25、50 步存 checkpoint，最多保留一个。输出位于共享盘 `artifacts/runs/a55-a56/train-real50-01/navsim`，包括 checkpoint 和最终模型。保存 ZeRO-3 checkpoint 前检查共享盘空间；两台必须访问同一个输出目录。换运行 ID 开新实验，沿用运行 ID 则复用原有目录。宿主机本地日志分别保存在各自的 `runs/a55-a56` 下。
