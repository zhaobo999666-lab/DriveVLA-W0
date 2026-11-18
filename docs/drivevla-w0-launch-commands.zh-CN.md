# DriveVLA-W0：容器和训练启动命令

下面的命令在 **A5 Linux 宿主机**执行，覆盖单机 8 卡、双机 16 卡和四机 32 卡。前提是镜像已构建或导入，本机工作区与共享盘已经准备好。创建容器只是把 NPU 和文件挂进去；模型、数据、依赖与代码发布仍按[单机部署指南](drivevla-w0-qwen3vl-a5-single-node-from-zero.zh-CN.md)或[双机部署指南](drivevla-w0-qwen3vl-a5-dual-node-from-zero.zh-CN.md)准备。

先在每台机器打开一个 Bash 终端，加载本机配置：

```bash
source /data1/zhaobo/a5-env.sh
export ZB_NODE="${ZB_NODE:-${A5_NODE:?请设置本机工作区对应的节点短名}}"
export ZB_RELEASE_DIR="${ZB_RELEASE_DIR:-$ZB_SHARED/artifacts/code/qwen3vl-dual}"
```

`ZB_NODE` 必须与已部署的容器和工作区后缀一致。训练启动器按 `zhaobo-drivevla-$ZB_NODE` 找容器，按 `/data1/zhaobo/drivevla-w0-$ZB_NODE` 找本机工作区。上面的 `ZB_RELEASE_DIR` 默认值是双机指南使用的共享发布路径；**如果当时发布在别处，在运行训练前改成实际目录**。该目录应已包含 `start-training.sh`、发布副本和 `REVISION`。

## 1. 单机启动容器：在一台机器上执行

如果自己的容器已经建好、只是停了，直接启动；原来的设备挂载和共享内存设置会保留：

```bash
docker start "$ZB_CONTAINER"
```

首次创建容器时，用下面这段。镜像名取自刚才加载的 `ZB_IMAGE`，例如部署指南里的 `zhaobo/drivevla-w0:qwen3vl-cann9.2-torch2.11-py312`。如果当地镜像叫别的名字，改变量即可。

```bash
# 用子 shell，检查失败时只退出这段命令。
(
  set -euo pipefail
  mountpoint -q /data1
  mountpoint -q /mnt/sfs_turbo
  test -d "$ZB_HOME"
  docker image inspect "$ZB_IMAGE" >/dev/null

  DEVICES=()
  for dev in /dev/davinci{0..7} /dev/davinci_manager /dev/hisi_hdc; do
    test -e "$dev" || { echo "缺少设备节点：$dev"; exit 1; }
    DEVICES+=(--device "$dev")
  done
  if [[ -e /dev/devmm_svm ]]; then
    DEVICES+=(--device /dev/devmm_svm)
  fi

  MOUNTS=()
  for src in /usr/local/Ascend/driver /usr/local/Ascend/firmware \
    /usr/local/dcmi /usr/local/sbin/npu-smi /usr/bin/hccn_tool \
    /etc/ascend_install.info; do
    if [[ -e "$src" ]]; then
      MOUNTS+=(--mount "type=bind,src=$src,dst=$src,readonly")
    fi
  done

  docker run -dit --name "$ZB_CONTAINER" \
    --label owner=zhaobo \
    --network host \
    --shm-size="${SHM_SIZE:-64g}" \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864 \
    "${DEVICES[@]}" \
    "${MOUNTS[@]}" \
    --mount "type=bind,src=$ZB_HOME,dst=/work/zhaobo" \
    --mount "type=bind,src=$ZB_SHARED/models,dst=/work/zhaobo/models" \
    --mount "type=bind,src=$ZB_SHARED/downloads,dst=/work/zhaobo/downloads" \
    --mount "type=bind,src=$ZB_SHARED/data,dst=/work/zhaobo/data" \
    --mount "type=bind,src=$ZB_SHARED/artifacts,dst=/work/zhaobo/artifacts" \
    -w /work/zhaobo \
    "$ZB_IMAGE" sleep infinity
)
```

| 参数 | 管什么 |
|---|---|
| `-dit` | `-d` 后台运行；`-i` 保持标准输入打开；`-t` 分配终端。 |
| `--name "$ZB_CONTAINER"`、`--label owner=zhaobo` | 设置容器名字和归属标签。训练脚本按名字找到容器。 |
| `--network host` | 容器直接使用宿主机网络。多机训练通过各自的 HCCL 网卡通信，不需要再用 `-p` 映射端口。 |
| `--shm-size` | 容器自己的 `/dev/shm` 上限，主要供 DataLoader 等进程共享数据。指南用 64 GiB；复现此前扩容后的配置，可在创建前 `export SHM_SIZE=96g`。它跟 NPU HBM 是两回事。 |
| `--ulimit memlock=-1:-1`、`--ulimit stack=67108864` | 放开内存锁定的软、硬限制；设置栈大小限制为 64 MiB。 |
| `DEVICES` 里的 `--device` | 把本机 8 张卡的设备节点及管理、通信相关设备交给容器。 |
| `MOUNTS` 里的 `--mount …,readonly` | 只读挂入宿主机已有的驱动、固件和诊断工具，容器内外路径相同。 |
| 其余 `--mount type=bind,src=…,dst=…` | `src` 是宿主机路径，`dst` 是容器内路径。本地工作区放代码和本机日志，模型、数据、下载缓存、产物放共享盘。共享挂载应事先存在。 |
| `-w /work/zhaobo`、`"$ZB_IMAGE"`、`sleep infinity` | 设置容器默认工作目录、指定镜像、让容器常驻。启动训练时，脚本再用 `docker exec` 在这个容器里运行。 |

## 2. 多机启动容器：每台机器各执行一次

双机要起 **2 个容器**，四机要起 **4 个容器**，每台都只挂本机的 8 张卡。首次建容器，在各宿主机加载各自的 `a5-env.sh` 后，**各执行一次第 1 节的 `docker run` 整段**，所有 Docker 参数含义相同。容器已建好的话，各台只需执行：

```bash
source /data1/zhaobo/a5-env.sh
docker start "$ZB_CONTAINER"
```

每台检查一次。第二行输出的镜像 ID 在所有节点上应一致：

```bash
docker inspect --format '{{.State.Running}} {{.HostConfig.NetworkMode}}' "$ZB_CONTAINER"
docker inspect --format '{{.Image}}' "$ZB_CONTAINER"
docker exec "$ZB_CONTAINER" df -h /dev/shm
docker exec "$ZB_CONTAINER" npu-smi info
```

`--format` 指定只打印需要的字段；`docker exec` 在运行中的容器里执行后面的命令。每台的 `ZB_HOME`、`ZB_CONTAINER` 对应本机，镜像 ID 和共享盘资源应该一致。容器命令里不用设置 `nnodes`、`node_rank` 或 `master_addr`，这些是下一步启动训练时用的。

## 3. 单机启动训练：一台 8 卡

同一台宿主机按顺序跑。用正式 trainval 之前，确认 pkl 与图片根目录是同一套数据：

```bash
cd "$ZB_RELEASE_DIR" || exit 1
export ZB_DATA=/work/zhaobo/data/prepared-trainval/train.pkl
export ZB_DATA_ROOT=/work/zhaobo/data/raw-camera/openscene-v1.1/sensor_blobs/trainval

bash start-training.sh check single-check-01 fast 24 qwen3vl 1
bash start-training.sh smoke single-smoke-01 fast 24 qwen3vl 1
bash start-training.sh train single-train-01 fast 24 qwen3vl 1
```

每条确认成功后再跑下一条。首次部署可以先用 `safe 2` 做 3 步冒烟；切回 `fast` 后，仍要验 `fast` 的通信和冒烟。若遇到复盘中的 `aclnnAmax 561000`，先按 FAQ 确认兼容层；下面的性能入口会为 fast ZeRO-2 使用它。

| 位置或变量 | 含义 |
|---|---|
| `check / smoke / train` | 验 HCCL 集合通信 / 跑 3 步且不保存权重 / 跑 50 步并保存权重和 checkpoint。 |
| `single-train-01` | 本次运行的名字，会进入日志和输出目录。重跑换新名字。 |
| `fast` | FA2、ZeRO-2、启用 pin memory、2 个 DataLoader worker；`safe` 用 SDPA、ZeRO-3 CPU offload、关闭 pin memory、0 个 worker。 |
| `24` | 每卡 batch（MBS），即每张卡一次吃 24 条样本；不是所有卡加起来 24。 |
| `qwen3vl` | 选择本地 Qwen3-VL-8B-Instruct 权重。ROSS 使用共享盘挂进容器的 SD1.5 权重。 |
| 最后的 `1` | 一台宿主机，固定 node rank 0，用本机 HCCL 网卡地址，启动 8 个训练进程。 |
| `ZB_DATA`、`ZB_DATA_ROOT` | 容器内的 pkl 文件和对应的图片根目录。pkl 里的图片路径相对于这个根目录。 |
| `ZB_RELEASE_DIR` | 宿主机共享盘上的代码发布目录。两台机器必须使用同一发布版本。 |

## 4. 多机启动训练：两台 16 卡，或四台 32 卡

**双机保存权重的训练**用同一个 `start-training.sh`。在两台机器都加载各自的环境，执行第 3 节的 `cd` 和两个数据路径设置，再分别设置本机编号：

```bash
# 第一台
export ZB_NODE_RANK=0
export ZB_PAIR=drivevla-dual-01
```

```bash
# 第二台
export ZB_NODE_RANK=1
export ZB_PAIR=drivevla-dual-01
```

以下三条命令，在两台分别执行。**每一阶段先在 rank 0 执行，再启动 rank 1；两边都结束后才进下一阶段**。不要在一台上跑了三个阶段才去另一台。

```bash
bash start-training.sh check dual-check-01 fast 24 qwen3vl 2
bash start-training.sh smoke dual-smoke-01 fast 24 qwen3vl 2
bash start-training.sh train dual-train-01 fast 24 qwen3vl 2
```

新增的是 `ZB_NODE_RANK` 和 `ZB_PAIR`。前者是**机器编号**，不是卡编号；后者是两台共用的配对名字。rank 0 会把自己的 HCCL 地址写到共享盘，rank 1 根据相同的配对名和本阶段运行名读取它。只有需要手动指定时，才在 rank 1 设置 `ZB_MASTER_ADDR` 为 rank 0 的 HCCL 地址。启动器对预置节点可能已有固定编号，选节点时按实际发布配置核对。最后的 `2` 表示两台机器，其余位置参数与单机相同。

**双机、四机做 50 步性能测试**用 `benchmark-mfu.sh`。四机不能把 `start-training.sh` 最后的数字改成 `4`：这个入口只接受 `1` 和 `2`。先在每台宿主机把 `PERF_DIR` 指向已经发布到共享盘的性能目录（里面有 `benchmark-mfu.sh`、计数脚本和 ZeRO-2 兼容入口）。再设置相同的节点列表和运行名，`ZB_NODE` 仍是各机器自己的值：

```bash
# 双机：两台填同一份列表和运行名
export ZB_SCALE_NODES=2
export ZB_NODE_LIST='<rank0短名>,<rank1短名>'
export ZB_MULTI_RUN=dual-measure-01

```

```bash
# 四机：用这一组代替双机设置；四台都设成一样
export ZB_SCALE_NODES=4
export ZB_NODE_LIST='<rank0短名>,<rank1短名>,<rank2短名>,<rank3短名>'
export ZB_MULTI_RUN=quad-measure-01
```

```bash
# 两台或四台分别执行，先启动列表中的第一台
: "${PERF_DIR:?请设置共享盘上的已发布性能目录}"
cd "$PERF_DIR" || exit 1
unset ZB_PAIR_RUN ZB_SINGLE_RUN
ZB_JOB=measure ZB_MBS=24 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10 \
ZB_FUSED_GATEUP=1 ZB_GC_SKIP_TEXT_LAST=2 ZB_FA2_UNPAD_CACHE=1 \
bash benchmark-mfu.sh
```

列表里的占位值要换成实际节点短名，逗号后不加空格。性能脚本按列表顺序计算 rank，不读取上面传统入口的 `ZB_NODE_RANK` 来定 rank。节点若不在脚本预设的地址表里，需在各台补同一份 `ZB_NODE_IP_<节点短名大写>`，值是各节点 HCCL 网卡的实际 IPv4；短名里的连字符改成下划线。例如短名为 `node1` 时，变量名是 `ZB_NODE_IP_NODE1`。正式启动前，可给上面最后一行加 `--dry-run` 核对发布目录、节点、rank、地址和配置。

| 性能启动参数 | 含义 |
|---|---|
| `PERF_DIR` | 宿主机共享盘上的性能脚本目录。与 `ZB_RELEASE_DIR` 是不同用途：一个放性能启动器及其依赖，一个放训练发布版本。 |
| `ZB_SCALE_NODES=2/4`、`ZB_NODE_LIST` | 机器数量和固定顺序。第一台是 rank 0；每台都必须出现在同一份列表里。 |
| `ZB_MULTI_RUN` | 同一次多机性能实验的名字，各台一致。已有结果时使用新名字。旧双机写法 `ZB_PAIR_RUN` 也受支持，两者不能同时设置。 |
| `ZB_JOB=measure` | 正式测速，关闭 Profiler，不保存权重。脚本用预设的 Qwen3-VL + ROSS、trainval、fast 配置。 |
| `ZB_MBS=24` | 每卡 batch。脚本梯度累积固定为 1。 |
| `ZB_MFU_STEPS=50` | 总共做 50 次优化器更新。 |
| `ZB_MFU_WARMUP=10` | 启动器传给训练的学习率 warmup 步数，同时用它跳过测速汇总的前 10 步，稳态统计第 11–50 步。 |
| `ZB_FUSED_GATEUP=1` | 开启文本 MLP 的 gate/up 融合。 |
| `ZB_GC_SKIP_TEXT_LAST=2` | 文本最后两层不做激活重算，用额外显存换少一点重复计算。 |
| `ZB_FA2_UNPAD_CACHE=1` | 在同一 microbatch 内复用 FA2 去 padding 所需的索引等信息。这组开关要求发布版本有对应代码。 |

要跑**同口径的单机 50 步测速**，在一台宿主机上执行：

```bash
: "${PERF_DIR:?请设置共享盘上的已发布性能目录}"
cd "$PERF_DIR" || exit 1
unset ZB_MULTI_RUN ZB_PAIR_RUN ZB_SCALE_NODES ZB_NODE_LIST
ZB_SINGLE_RUN=single-measure-01 \
ZB_JOB=measure ZB_MBS=24 ZB_MFU_STEPS=50 ZB_MFU_WARMUP=10 \
ZB_FUSED_GATEUP=1 ZB_GC_SKIP_TEXT_LAST=2 ZB_FA2_UNPAD_CACHE=1 \
bash benchmark-mfu.sh
```

`ZB_SINGLE_RUN` 给单机实验命名，并启用与多机对比用的墙钟计时。需要采 Insight 时，另换运行名，用 `ZB_JOB=profile`，保留 50 步；要采整段再加 `ZB_PROFILE_FULL=1`。单独的 Profiler 轮用于看耗时，不用它替代正式测速。

### 日志里的底层参数怎么读

| 底层参数 | 单机 / 双机 / 四机 | 含义 |
|---|---|---|
| `--nnodes` | `1 / 2 / 4` | 宿主机数量。 |
| `--node_rank` | `0` / `0,1` / `0,1,2,3` | 机器在本作业里的编号。同一台上的 8 个训练进程另有各自的 rank。 |
| `--ngpus` → `torchrun --nproc_per_node` | 都是 `8` | 每台启动 8 个训练进程，一卡一个。总卡数分别是 8、16、32。 |
| `--master_addr`、`--master_port` | rank 0 的 HCCL 地址；训练 `29555`，普通脚本的 `check` 用 `29556` | 各节点到同一地址、同一端口会合。容器走 host 网络，端口需在运行前空闲。 |
| `--batch_size`、`--grad_accum` | 本例每卡 `24`，累积 `1` | 全局 batch = 每卡 batch × 总卡数 × 累积步数。本例分别是 **192、384、768** 条样本/优化器步。 |
| `--max_steps`、`--logging_steps` | 普通 `train` 和本例 `measure` 都是 `50` 步、每 `1` 步记一次日志 | step 指优化器更新；打印日志的频率和更新频率是两件事。 |
| `--warmup_steps` | 普通 `train` 为 `5`；上面的测速为 `10` | 学习率预热步数。稳态统计窗口还要看对应汇总脚本的设置。 |
| `--device`、`--fp`、`--attn_type`、`--deepspeed_config` | 本例为 `npu`、`bf16`、`fa2`、ZeRO-2 配置 | 执行设备、数值精度、注意力实现和 DeepSpeed 并行配置。 |
| `--model_name_or_path`、`--sd_model_path` | 容器内 Qwen3-VL 和 SD1.5 权重目录 | 前者是训练骨干，后者供 ROSS 加载 UNet/VAE 等组件。 |
| `--data_path`、`--data_root` | 容器内的 trainval pkl 和图片根目录 | pkl 提供样本信息，图片相对路径在图片根目录下解析。 |
| `--num_workers`、`--pin_memory` | fast 为 `2`、`true`；safe 为 `0`、`false` | 分别控制每个训练进程的取数 worker 数量、是否尝试用页锁定的主机内存准备数据。 |
| `--max_pixels`、`--min_pixels`、`--model_max_length`、`--seed` | 性能脚本为 `903168`、`12544`、`4096`、`42` | 控制图片预处理的像素上下限、模型序列长度上限和随机种子。改图片大小或序列长度会改变实际工作量。 |
| `--save_model`、`--save_steps`、`--save_total_limit` | 普通 `train`：`true`、`25`、`1`；测速不保存模型 | 控制是否写模型、每隔多少步保存 checkpoint、保留几份 checkpoint。 |
| `--report_to` | 普通入口为 `tensorboard`，性能入口为 `none` | 控制向哪个训练日志后端上报。`none` 仍保留启动器的文本日志。 |
| `--npu_profiling`、`--npu_profiler_level 1` | 只有性能入口的 `profile` 轮加上 | 启用 NPU Profiler 并指定采集级别。日常说的 profiling，在这个启动器里对应 `ZB_JOB=profile`。 |
| `--output_dir`、`--exp_name` | 启动器按运行名生成目录；实验子目录是 `navsim` | 去哪里找日志、训练状态及保存产物。 |

与启动有关的问题可对照[真实问题复盘](drivevla-w0-deployment-training-common-issues.zh-CN.md)和[通用 FAQ](drivevla-w0-deployment-training-faq.zh-CN.md)。
