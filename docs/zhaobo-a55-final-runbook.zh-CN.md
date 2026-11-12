# zhaobo｜A55 单机 DriveVLA-W0 从零实施手册

更新：2026-11-09（A55 实施，共享盘保存可复用资源）。入口：`ssh ott9-a55`。已登录核实主机名 `liteserver-a5-2-00001.novalocal`、ARM64、Huawei Cloud EulerOS 3.0 和 8 张 Ascend 950DT。

**目前先备料，等卡空闲再训练。** 11 月 9 日约 10:03（北京时间）检查时，8 张卡均被 `VLLMWorker_DP` 占用，每卡总 HBM 使用约 96,045–96,070 / 98,304 MB。即使利用率显示 0%，也不能视为空闲。可先做第 2–5 节的镜像、代码和模型准备，以及第 8 节的数据下载与预处理；第 6、7、9、10 节涉及上卡的操作需等到设备可用。

目标：在自己的目录和容器中，从公开来源重新取得代码、环境、模型和数据，完成 **Qwen2.5-VL-7B + ROSS 的训练、保存、NPU 推理、轨迹与误差查看**，再扩展到 NAVSIM PDMS。参考机已有资源只用于调查，不作为下载或运行前提。

本文是当前对话的最终实施入口，替代此前通用的 A5 规划和依赖现成容器的运行说明。概念可参考原教程，路径、版本与命令以本手册为准。

### 已在 A55 发起准备：先查状态

11 月 9 日已创建本地工作区、上传固定代码 bundle，并启动环境、模型、数据三项独立准备任务。初次构建的 PyPI 文件下载超时，已用 A55 可达的公开 PyPI 镜像重试，固定的软件版本保持不变。 PyTorch CPU 索引解析过慢时，A55 的构建改用官方同版本 ARM64 wheel 直链及 SHA256 校验。**若准备任务仍在运行，不要同时手动重复下载。**

```bash
ssh ott9-a55 'cat /data1/zhaobo/drivevla-w0-a55/manifests/prepare-{environment,models,dataset}.status'
```

三项状态都是 `READY` 后，再从第 6 节做 NPU 验收。相应日志在同一目录的 `prepare-*.log`。环境准备包含依赖导入与训练入口帮助检查；模型准备包含分片与 SHA256 检查；数据准备包含归档解包、独立 train/val 划分及图片检查。准备脚本不会自动启动训练。

**可复用资源的宿主机路径是 `/mnt/sfs_turbo/zhaobo/drivevla-w0`**。模型权重、原始数据归档、解包后的数据、生成的训练／验证集以及确认可复用的归档都放这里。A55 的本地盘保留环境、日志、缓存和正在训练的 checkpoint。换机器时重新挂载同一共享目录，容器内的权重和数据路径保持不变。

## 资源准备速查：代码写了什么，自己要准备什么

**代码通常只指定资源的读取位置，下载和预处理需要另做。** 例如，Qwen 启动脚本给了默认骨干模型 ID，但要求使用者传入 `--data_path`、`--data_root`、`--sd_model_path`，然后检查文件和目录是否存在。默认路径出现在脚本里，并不表示文件随 Git 下载。

本手册的入口是 [train_qwen_vla_navsim.sh](../scripts/scripts_train/train_qwen_vla_navsim.sh)，使用 **Qwen2.5-VL + ROSS + RGB 图像**。刚分析的 `train_base_ar_withou_moe.sh` 使用 **Emu3 + VQ 图像码**，资源见下方对照。

| 资源 | 代码如何指定 | 从零如何取得、放在哪里 | 最迟准备时间 |
|---|---|---|---|
| CANN 镜像 | 训练脚本不拉镜像；本手册 [Dockerfile](zhaobo-a55/Dockerfile) 的 `FROM` 固定 digest | 从华为云 SWR 拉 ARM64 基础镜像，按第 3 节构建自己的训练镜像 | 建容器前 |
| NPU 驱动、固件、设备 | Docker 挂载宿主驱动并透传设备，容器内用 `torch_npu` | 宿主侧由维护方提供；容器侧 CANN 随镜像取得。第 2、6 节检查设备、版本与通信 | 训练前 |
| Python 环境 | 仓库有多套依赖清单；模型代码导入 torch、DeepSpeed、Diffusers 等 | 第 3 节 Dockerfile 安装 Python 3.12 配套依赖，第 4 节安装仓库内 Transformers | 训练前 |
| 代码、FAST、DeepSpeed JSON | `--action_tokenizer_path` 默认 `configs/fast`；DeepSpeed JSON 也在仓库内 | 第 4 节克隆指定 Git 提交，两类文件随代码取得 | 训练前 |
| Qwen 骨干 | `--model_name_or_path`，默认 HF ID；加载模型、tokenizer 和图像处理器 | 第 5 节下载固定 revision 到 `/work/zhaobo/models/Qwen2.5-VL-7B-Instruct`，显式传本地目录 | 合成测试前 |
| ROSS 的 SD1.5 组件 | `--sd_model_path`；训练读取 UNet、VAE 与 scheduler | 第 5 节下载指定组件到 `/work/zhaobo/models/stable-diffusion-v1-5` | 合成测试前 |
| OpenScene mini 原始日志与 RGB | `--data_root` 读取解包后的图片目录 | 第 8 节把 mini 日志和相机分片下载到 `/work/zhaobo/downloads/openscene`，解包到 `data/raw-logs`、`data/raw-camera` | 真实数据训练前 |
| 训练／留出 pickle、归一化、token 清单 | `--data_path` 读取训练 pickle；推理另读留出集和归一化文件 | 第 8 节运行 `support.py prepare`，从日志和 RGB 生成到 `/work/zhaobo/data/prepared` | 真实数据训练／推理前 |
| DriveVLA 驾驶 checkpoint | 推理接收 checkpoint 路径 | 第 9 节训练产生自己的 `real50/navsim`；本流程不要求下载作者的驾驶 checkpoint | 推理前 |
| nuPlan 地图、评估环境 | PDMS 读取地图及独立 NAVSIM 依赖 | 按第 12 节下载地图到 `/work/zhaobo/data/maps` 并建评估环境 | 做 PDMS 时 |

按这个顺序准备：**镜像和环境 → 固定代码 → Qwen 与 SD1.5 权重 → 单卡／八卡合成测试 → OpenScene mini → 生成 pickle → 真实训练 → 用自己的权重推理**。先不用下载 FAST 原版、Emu3 权重、全量 OpenScene 或 PDMS 地图。[资源清单](zhaobo-a55/resources.lock.json)用于核对来源和版本，不是训练脚本会读取的配置。

### Emu3 脚本的资源对照

| 项目 | `train_base_ar_withou_moe.sh` 使用的资源 |
|---|---|
| 骨干 | `--model_name_or_path` 默认指向 `pretrained_models/Emu3-Stage1`。自行下载 [BAAI/Emu3-Stage1](https://huggingface.co/BAAI/Emu3-Stage1)，也可以传入下载好的本地目录。 |
| 训练数据 | `--data_path` 默认指向 NavSim `*_trainval_pre_1s.pkl`；[数据集代码](../utils/datasets.py) 还会读取 pickle 中的 VQ `.npy` 文件。Qwen 路线的 RGB 和训练 pickle 不能直接替换它们。 |
| 数据来源 | [SETUP](../SETUP.md) 列出了作者 [liyingyan/DriveVLA-W0](https://huggingface.co/liyingyan/DriveVLA-W0) 的训练／测试 pickle 和 VQ ZIP；这次未能核实指定文件的公开下载可用性。若已取得可信的 ZIP 和 pickle，要将其中的图片路径改到自己的解包目录。若从原始数据重做，参照 [Training.md](../Training.md) 先生成 VQ 码，再生成配套 metadata。 |
| 视觉和动作 tokenizer | 已有 VQ 时，训练只需代码中的 `configs/fast` 作为动作 tokenizer。自己生成 VQ 或走默认推理链时，再准备 [BAAI/Emu3-VisionTokenizer](https://huggingface.co/BAAI/Emu3-VisionTokenizer)；默认推理还从模型根目录读取 `physical-intelligence-fast`。 |
| 环境 | [SETUP](../SETUP.md) 和 [requirements_ascend.txt](../requirements_ascend.txt) 是另一套环境参考。当前 Python 入口还导入 `msprobe`，应单独核对兼容版本。本文的 A55 Qwen 环境尚未验收 Emu3。 |

[setup_and_run_npu.sh](../scripts/setup_and_run_npu.sh) 假设 ZIP 和 pickle 已经下载，负责解包和改路径。它当前写的是 `train_vp_codes.zip`，与 SETUP 中的 `train_vq_codes.zip` 不同；它也使用 `apt-get`，不要直接照搬到本文的 openEuler 镜像。

## 0. 已确定的配置：不要再猜镜像、分支或数据路径

| 项目 | 固定选择 |
|---|---|
| 执行机器 | A55：`ssh ott9-a55` |
| 隔离名称 | `zhaobo`，不创建或假设已有 Linux 同名账号 |
| 宿主根目录 | `/data1/zhaobo/drivevla-w0-a55` |
| 共享资源根目录 | `/mnt/sfs_turbo/zhaobo/drivevla-w0` |
| 容器根目录 | `/work/zhaobo` |
| 容器 / tmux 名 | `zhaobo-drivevla-a55` |
| 自建镜像标签 | `zhaobo/drivevla-w0:a55-cann9.2-torch2.11-py312-v1` |
| 代码仓库 | `https://github.com/mingchang93/DriveVLA-W0.git` |
| 分支 | `qwen3vl-backbone`，此分支也支持 Qwen2.5；本教程实际加载 2.5-VL-7B |
| 固定代码提交 | `2a87094e0ac52c425004eacdcced8734b3f68938` |
| 通信端口 | 训练/推理 29555，独立通信测试 29556；执行前检查端口未占用 |
| 训练路线 | BF16、SDPA、每卡 batch 2、ZeRO-3 CPU offload |
| 首批真实数据 | OpenScene mini 原始日志 + mini 相机分片 0；不足则补分片 1、2 |
| 训练 metadata | 用本手册工具从原始日志生成，按 log 划分 train/val，仅训练集拟合归一化 |
| 最初结果 | 真实留出集的动作 JSON、loss 曲线、轨迹对比图、ADE/FDE |

**此前检查的 `main` 提交 `a55766ce…` 下没有 Qwen 训练启动脚本。** 请按第 4 节固定分支和提交，避免拿到不同版本的入口。

### 0.1 基础镜像已核实到 ARM64 digest

镜像来源：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-950-openeuler24.03-py3.12
```

**实际拉取使用这个不可变的 ARM64 manifest：**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann@sha256:f81ad0b1f1e9b57e7859a334a9b63ba4444f6af1f0721e87300922627756ec1c
```

11 月 9 日已在 A55 重新运行 `docker manifest inspect --verbose`，确认该 digest 对应 `linux/arm64`；此次只查询了 manifest。基础镜像使用 openEuler 24.03、CANN 9.2.0、Python 3.12，包管理器为 `dnf`。

参考任务的 `drivevla-w0:cann9.2.0-pytorch2.11.0-python3.12-a5` 是本地标签。此前读取的 `deploy_container.sh` 写明：本地没有该镜像时，从 `/mnt/sfs_turbo/zjj/images/drivevla-w0.tar` 执行 `docker load`。这说明了部署脚本预期的获取方式；tar 当前是否存在、镜像最初从哪里取得，仍未确认。

[Ascend Docker Hub](https://hub.docker.com/r/ascendai/torch-npu) 另有公开标签 `ascendai/torch-npu:2.11.0.post3-cann9.2.0-950-openeuler24.03-py3.12`。本次核对了发布者的标签列表，尚未验证该镜像在 A55 上的 manifest、实际拉取或训练。它不是已确认的参考镜像原件。本文继续使用上面固定 digest 的 SWR 路线，不需要给参考镜像重新打标签。

### 0.2 Python 框架基线

- PyTorch 2.11.0+cpu、torchvision 0.26.0+cpu、torchaudio 2.11.0+cpu。
- torch_npu 2.11.0.post3，已从 PyPI 元数据确认存在 CPython3.12 / aarch64 wheel，并记录 SHA256。
- DeepSpeed 0.19.7；训练依赖版本见附录 Dockerfile。
- 项目自己的 Transformers 由固定 Git 提交安装，不用 PyPI 的 4.44.0 代替。

这些组合依据 [Ascend 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)选择，并接近参考训练环境；**新镜像仍须实际构建和逐级验收，本次没有声称它已经训练通过。**

[requirements_qwen.txt](../requirements_qwen.txt) 记录的是 CANN 9.1.0／torch_npu 2.11.0 的另一套组合。不要在本手册的 CANN 9.2.0／torch_npu 2.11.0.post3 环境里整份覆盖安装；以 Dockerfile 和第 4 节为准。部分传递依赖由 pip 解析，安装后用 `pip-freeze.txt` 记录实际版本。

### 0.3 模型固定来源和 revision

| 用途 | 仓库 | revision |
|---|---|---|
| Qwen 骨干 | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| ROSS 的 UNet/VAE/scheduler | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| FAST 原始来源（可选对照） | `physical-intelligence/fast` | `ec4d7aa71691cac0b8bed6942be45684db2110f4` |
| 实际动作 tokenizer | 固定代码中的 `configs/fast/` | 随上述 Git commit 固定 |

来源：[Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)、[SD1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)、[FAST](https://huggingface.co/physical-intelligence/fast)。沿用此前已核对的 revision；A55 另测了 Qwen 固定版本的 `config.json`。Qwen 约 16.6 GB；SD1.5 只下载所需组件。

### 0.4 数据固定来源

```text
仓库：OpenDriveLab/OpenScene（dataset）
revision：a76f840b65e972bc45e56c2adced897498e9a026

openscene-v1.1/openscene_metadata_mini.tgz
openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_0.tgz
```

源：[OpenScene 文件目录](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/main/openscene-v1.1)。日志与 RGB 都重新下载；初次不用 LiDAR，也不用 Emu3 的 VQ 码。

作者 DriveVLA 资源仓库本次返回了大量权重/预测文件，但未能完整核实训练 pkl 的公开位置，后续 API 还返回了 429。本方案因此不依赖“作者某个 pkl 应该能下载”，直接从官方原始日志生成自己的数据。

### 0.5 网络结论

11 月 9 日在 A55 检查：SWR 固定镜像 manifest 可读；PyPI 的 DeepSpeed 索引和 PyTorch CPU wheel 索引返回 200。固定 Qwen `config.json` 通过 `hf-mirror.com` 返回 200，直连 `huggingface.co` 超时；GitHub 的 `git ls-remote` 在 15 秒内未完成。

第 4 节优先用本地 Git bundle，第 5、8 节默认用 `HF_ENDPOINT=https://hf-mirror.com`。这些检查只覆盖宿主机上的小请求，未验证容器内网络和大文件下载。下载前先试小文件；遇 429 降低并发、间隔后重试。资源身份仍以发布仓库和固定 revision 为准。

## 1. 本地运行目录与共享资源目录

zhaobo 隔离的是目录、容器、Python 环境、模型副本、缓存、端口和实验记录。**它不会独占 NPU、宿主内存或共享网络；使用 root 登录也不构成账号级安全隔离。** 运行前仍需确认设备使用时段，不停止其他人的任务。

```text
本地 /data1/zhaobo/drivevla-w0-a55  →  容器 /work/zhaobo
├── bootstrap/         Dockerfile、support.py、资源清单
├── code/DriveVLA-W0/  当前机器的 Git 工作树
├── tools/、scratch/  推理副本、通信验收脚本、合成测试数据
├── runs/、results/    当前任务的日志、checkpoint、预测与指标
└── cache/、envs/、manifests/  本机缓存、环境和验收记录

共享 /mnt/sfs_turbo/zhaobo/drivevla-w0
├── models/     → /work/zhaobo/models      Qwen、SD1.5、已验收的训练权重
├── downloads/  → /work/zhaobo/downloads   OpenScene、地图的原始归档
├── data/       → /work/zhaobo/data        raw-logs、raw-camera、prepared、maps
└── artifacts/  → /work/zhaobo/artifacts   可复用 Git bundle、验收后的镜像 tar
```

`/data1` 是 A55 本地 ext4 盘，检查时剩余约 **3.7 TiB**，Docker 数据根目录在 `/data1/docker`。`/mnt/sfs_turbo` 是共享 NFS，检查时文件系统剩余约 **13 TiB**（不等于个人配额）。只使用自己的项目子目录，不依赖 A57 的本地文件。共享资源由一台机器下载、解包和生成，验收完成后再给其他机器复用；不要并行写同一目录。

## 2. A55 宿主机：创建自己的空工作区

以下逐块执行，注意每块标注的运行位置。

```bash
# [自己的电脑]
ssh ott9-a55
```

```bash
# [A55 宿主机]
export ZB_HOME=/data1/zhaobo/drivevla-w0-a55
export ZB_SHARED=/mnt/sfs_turbo/zhaobo/drivevla-w0
export ZB_CONTAINER=zhaobo-drivevla-a55
export ZB_IMAGE=zhaobo/drivevla-w0:a55-cann9.2-torch2.11-py312-v1

test "$(hostname)" = liteserver-a5-2-00001.novalocal || { echo '主机名与手册记录不同，请先核对 SSH 入口'; exit 1; }
uname -m
npu-smi info
df -h /data1 /mnt/sfs_turbo
docker version
ss -lnt '( sport = :29555 or sport = :29556 )'

# 先确认挂载和写权限，避免 NFS 未挂载时写进宿主机根盘。
mountpoint -q /mnt/sfs_turbo || { echo '共享盘未挂载'; exit 1; }
test -w /mnt/sfs_turbo/zhaobo || { echo '共享目录不可写'; exit 1; }
test "$(realpath -m "$ZB_SHARED")" = "$ZB_SHARED" || { echo '共享项目路径指向其他位置'; exit 1; }
mkdir -p "$ZB_SHARED"/{models,downloads,data/raw-logs,data/raw-camera,data/maps,artifacts/code,artifacts/images}

# 第一次必须是新目录；若已存在，先检查是否为自己的上次结果，不要清空重来。
test ! -e "$ZB_HOME" || { echo '目标目录已存在，请先核对'; exit 1; }
mkdir -p "$ZB_HOME"/{bootstrap,code,tools,scratch,runs,results,cache,envs,manifests}

{ hostname; uname -a; cat /etc/os-release; npu-smi info; } > "$ZB_HOME/manifests/machine.txt"
```

两个端口在此次检查中均无监听，执行前仍需重查。A55 已有参考训练镜像和 vLLM 镜像，从零流程不依赖它们。

本教程不覆盖或升级共享服务器的宿主驱动和固件；它们已由服务器维护方部署。裸机安装由管理员先按硬件配套完成，再进入本步骤。

## 3. 基础镜像 → 自己的训练镜像 → 新容器

### 3.1 准备引导文件

本次交付附带 `Dockerfile`、`support.py`、`resources.lock.json`。它们是本次文档的辅助文件，**不在指定的远程 Git commit 中**。可从交付包取出，或把本文末尾附录的完整文本保存为：

```text
$ZB_HOME/bootstrap/Dockerfile
$ZB_HOME/bootstrap/support.py
$ZB_HOME/bootstrap/resources.lock.json   # 可选保存，核心命令中的固定值已完整列出
```

先执行文末附录的两个文件创建命令，再返回第 3.2 节。它们只在自己的 bootstrap 目录写入文件，不覆盖 clone 出来的源码。若以后要修改项目核心代码，应在自己的受控分支管理并记录提交；本次没有自动 commit/push。

### 3.2 拉取与构建

```bash
# [A55 宿主机]
set -o pipefail
docker pull --platform linux/arm64 \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/cann@sha256:f81ad0b1f1e9b57e7859a334a9b63ba4444f6af1f0721e87300922627756ec1c

docker build --platform linux/arm64 \
  -t "$ZB_IMAGE" "$ZB_HOME/bootstrap" \
  2>&1 | tee "$ZB_HOME/manifests/docker-build.log"

docker image inspect "$ZB_IMAGE" > "$ZB_HOME/manifests/image.json"
```

镜像构建会安装编译工具、独立训练 venv、torch/NPU 和项目层依赖；torch_npu wheel 下载后按已核实 SHA256 校验。构建期间会联网下载软件，但不加载模型、不使用 NPU 训练。

若 `dnf` 仓库、PyTorch CPU 源或 PyPI 不通，构建将在对应行停止；先解决该下载来源，不换成未经核对的 CUDA 包或机器上的旧镜像。

### 3.3 容器设备与目录挂载

```bash
# [A55 宿主机] 以下节点已经只读核实存在
DEVICES=()
for dev in /dev/davinci{0..7} /dev/davinci_manager /dev/hisi_hdc; do
  test -e "$dev" || { echo "缺少设备 $dev"; exit 1; }
  DEVICES+=(--device "$dev")
done
# A55 本次未看到 devmm_svm，若维护后出现，再按实际情况加入。
[ ! -e /dev/devmm_svm ] || DEVICES+=(--device /dev/devmm_svm)

docker run -dit --name "$ZB_CONTAINER" \
  --network host --shm-size=16g --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  "${DEVICES[@]}" \
  -v "$ZB_HOME:/work/zhaobo" \
  --mount "type=bind,src=$ZB_SHARED/models,dst=/work/zhaobo/models" \
  --mount "type=bind,src=$ZB_SHARED/downloads,dst=/work/zhaobo/downloads" \
  --mount "type=bind,src=$ZB_SHARED/data,dst=/work/zhaobo/data" \
  --mount "type=bind,src=$ZB_SHARED/artifacts,dst=/work/zhaobo/artifacts" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -w /work/zhaobo "$ZB_IMAGE"

docker exec -it "$ZB_CONTAINER" bash --noprofile --norc
```

已有同名容器时，先检查 `docker inspect` 的 `Mounts`。旧容器若没有上述共享挂载，不要在里面下载资源；使用新配置另起自己的容器。下载和预处理需要读写挂载；之后在其他机器只训练或推理时，可以给共享挂载加 `,readonly`。

在宿主机运行以下命令，确认模型和数据的源路径以 `/mnt/sfs_turbo/zhaobo/drivevla-w0/` 开头，再开始下载：

```bash
docker inspect --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' "$ZB_CONTAINER"
```

## 4. 取得固定代码，安装仓库内 Transformers

A55 直连 GitHub 的检查超时，推荐用本地现有仓库的 Git bundle。它只打包 Git 对象和分支，可以核对精确提交。共享盘已有同一提交的可用 bundle 时，直接进入容器克隆。首次上传前先按第 2 节建好目录，再在自己的电脑上执行：

```powershell
# [自己的 Windows 电脑，PowerShell]
$ZbRepo = 'D:\OTT9\DriveVLA-W0'
$ZbCommit = '2a87094e0ac52c425004eacdcced8734b3f68938'
$ZbBundle = Join-Path $env:TEMP "DriveVLA-W0-$ZbCommit.bundle"
git -C $ZbRepo merge-base --is-ancestor $ZbCommit refs/heads/qwen3vl-backbone
if ($LASTEXITCODE -ne 0) { throw '本地分支不含固定提交，请先从指定 GitHub 仓库取得它' }
git -C $ZbRepo bundle create $ZbBundle refs/heads/qwen3vl-backbone
if ($LASTEXITCODE -ne 0) { throw '生成 Git bundle 失败' }
scp $ZbBundle "ott9-a55:/mnt/sfs_turbo/zhaobo/drivevla-w0/artifacts/code/DriveVLA-W0-$ZbCommit.bundle"
if ($LASTEXITCODE -ne 0) { throw '上传 Git bundle 失败' }
```

```bash
# [新容器]
export ZB_WORK=/work/zhaobo
source /opt/zhaobo/train/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd "$ZB_WORK/code"
ZB_COMMIT=2a87094e0ac52c425004eacdcced8734b3f68938
ZB_BUNDLE="$ZB_WORK/artifacts/code/DriveVLA-W0-$ZB_COMMIT.bundle"
if [ -s "$ZB_BUNDLE" ]; then
  git clone --branch qwen3vl-backbone --single-branch \
    "$ZB_BUNDLE" DriveVLA-W0
  git -C DriveVLA-W0 remote set-url origin https://github.com/mingchang93/DriveVLA-W0.git
else
  # 仅在 A55 能直连 GitHub 时使用。
  git clone --branch qwen3vl-backbone --single-branch \
    https://github.com/mingchang93/DriveVLA-W0.git
fi
cd DriveVLA-W0
git checkout --detach 2a87094e0ac52c425004eacdcced8734b3f68938
test "$(git rev-parse HEAD)" = 2a87094e0ac52c425004eacdcced8734b3f68938

python -m pip install -e ./reference/transformers
python -m pip check
python -m pip freeze > "$ZB_WORK/manifests/pip-freeze.txt"
git rev-parse HEAD > "$ZB_WORK/manifests/git-commit.txt"
```

然后创建每天进入容器后都能复用的环境文件：

```bash
cat > /work/zhaobo/bootstrap/enter-env.sh <<'SH'
export ZB_WORK=/work/zhaobo
source /opt/zhaobo/train/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$ZB_WORK/code/DriveVLA-W0"
export PYTHONPATH="$PWD:$PWD/reference/Qwen2.5-VL/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export DEVICE=npu
export INF_NAN_MODE_ENABLE=1
export ROSS_NPU_RMSNORM=0 ROSS_NPU_ROPE=0 ROSS_FUSED_GATEUP=0 ROSS_FUSED_SWIGLU=0 ROSS_FUSED_ADD_RMSNORM=0
SH
source /work/zhaobo/bootstrap/enter-env.sh

python -c 'import torch,torch_npu,transformers,deepspeed,diffusers; print(torch.__version__,torch_npu.__version__,transformers.__version__,transformers.__file__,deepspeed.__version__,diffusers.__version__)'
DEVICE=cpu python utils/train_qwen_vla.py --help > /work/zhaobo/manifests/train-help.txt
```

验收：Python 和 torchrun 来自 `/opt/zhaobo/train/bin`；Transformers 指向自己的 `/work/zhaobo/code/DriveVLA-W0/reference/transformers`；参数帮助正常打印。不要使用宿主机系统 Python 或别人的 venv。

## 5. 下载 Qwen 骨干与 SD1.5 组件到共享盘

先确认 `/work/zhaobo/models` 挂载到共享目录。已有相同 revision 的权重时，先做本节文件验收；校验通过则直接复用，不用重下。第一次下载先用一个小文件验证 endpoint。若返回 429，等待后再用单并发重试。

```bash
# [容器，已 source enter-env.sh]
hf download Qwen/Qwen2.5-VL-7B-Instruct config.json \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --local-dir /work/zhaobo/models/Qwen2.5-VL-7B-Instruct

hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 --max-workers 1 \
  --local-dir /work/zhaobo/models/Qwen2.5-VL-7B-Instruct

hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 --max-workers 1 \
  --include 'unet/config.json' 'unet/diffusion_pytorch_model.safetensors' \
            'vae/config.json' 'vae/diffusion_pytorch_model.safetensors' \
            'scheduler/scheduler_config.json' 'model_index.json' \
  --local-dir /work/zhaobo/models/stable-diffusion-v1-5
```

动作 tokenizer 直接使用固定代码的 `configs/fast`。如需对照上游实现，可从第 0.3 节的 FAST 固定 revision 单独下载到 `/work/zhaobo/models/fast-reference`，不覆盖项目自己的 tokenizer。

模型文件验收：

```bash
python - <<'PY'
import json
from pathlib import Path
q=Path('/work/zhaobo/models/Qwen2.5-VL-7B-Instruct')
idx=json.loads((q/'model.safetensors.index.json').read_text())
for name in sorted(set(idx['weight_map'].values())):
    p=q/name
    assert p.is_file() and p.stat().st_size > 1024*1024, p
for name in ['config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json']:
    assert (q/name).is_file(), name
s=Path('/work/zhaobo/models/stable-diffusion-v1-5')
for name in ['unet/config.json','unet/diffusion_pytorch_model.safetensors','vae/config.json','vae/diffusion_pytorch_model.safetensors','scheduler/scheduler_config.json']:
    assert (s/name).is_file(), name
fast=Path('/work/zhaobo/code/DriveVLA-W0/configs/fast')
for name in ['processor_config.json','tokenizer.json','tokenizer_config.json']:
    json.loads((fast/name).read_text(encoding='utf-8'))
assert (fast/'processing_action_tokenizer.py').is_file(), fast
print('Required model files present')
PY
(
  cd /work/zhaobo/models
  if [ -s base-models.sha256 ]; then
    sha256sum -c base-models.sha256
  else
    find Qwen2.5-VL-7B-Instruct stable-diffusion-v1-5 -type f -name '*.safetensors' \
      -exec sha256sum {} + > base-models.sha256
  fi
)
```

文件存在检查不等于模型能加载，后续最小训练和保存后的重加载继续验收。

## 6. 先验收 NPU 和单机通信

在宿主机重查 `npu-smi info`：先确认进程列表、HBM 占用和设备编号，再执行下面的测试。仍有 vLLM 占用时可先做第 8 节；不要仅凭利用率为 0% 判定设备空闲。

以下单卡示例使用物理卡 0，需确认该卡可用；八卡 HCCL 测试需等全部八张卡可用。

```bash
# [容器，卡 0 已空闲]
ASCEND_RT_VISIBLE_DEVICES=0 python - <<'PY'
import torch,torch_npu
print(torch.npu.is_available(),torch.npu.device_count())
x=torch.tensor([1.,2.,3.],device='npu:0')
assert (x*x).sum().cpu().item() == 14.0
print('NPU arithmetic OK')
PY

cat > /work/zhaobo/tools/check_hccl.py <<'PY'
import os
from datetime import timedelta
import torch,torch_npu
import torch.distributed as dist
r=int(os.environ['LOCAL_RANK'])
torch.npu.set_device(r)
dist.init_process_group('hccl', timeout=timedelta(seconds=120))
x=torch.tensor([dist.get_rank()+1.],device=f'npu:{r}')
dist.all_reduce(x)
expected=dist.get_world_size()*(dist.get_world_size()+1)/2
assert x.cpu().item()==expected
print('rank',r,'sum',x.cpu().item(),flush=True)
dist.destroy_process_group()
PY
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29556 \
  /work/zhaobo/tools/check_hccl.py
```

八个 rank 应都得到 36。单卡成功而这一步失败，先检查资源、设备绑定和 HCCL，不修改训练样本来排查通信问题。

## 7. 单卡合成数据 3 步 → 单机八卡 3 步

长任务进入自己的 tmux，避免 SSH 断开影响学习过程：

```bash
# [A55 宿主机，新终端]
tmux new -s zhaobo-drivevla-a55
docker exec -it zhaobo-drivevla-a55 bash --noprofile --norc
source /work/zhaobo/bootstrap/enter-env.sh
```

容器内生成合成数据，再执行单卡最小训练：

```bash
python tools/synth/synth_qwen_vla_data.py --out_root /work/zhaobo/scratch/synthetic --num_scenes 64
set -o pipefail
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/scripts_train/train_qwen_vla_navsim.sh \
  --model_name_or_path /work/zhaobo/models/Qwen2.5-VL-7B-Instruct \
  --sd_model_path /work/zhaobo/models/stable-diffusion-v1-5 \
  --data_path /work/zhaobo/scratch/synthetic/synth_navsim.pkl \
  --data_root /work/zhaobo/scratch/synthetic/images \
  --action_tokenizer_path "$PWD/configs/fast" \
  --device npu --attn_type sdpa --fp bf16 \
  --ngpus 1 --batch_size 2 --num_workers 0 --pin_memory false \
  --deepspeed_config "$PWD/scripts/sft/zero3_offload.json" \
  --max_pixels 150528 --min_pixels 12544 --use_previous_actions false \
  --max_steps 3 --warmup_steps 0 --logging_steps 1 --save_model false \
  --report_to none --master_port 29555 \
  --output_dir /work/zhaobo/runs/synth-single3 --exp_name navsim \
  2>&1 | tee /work/zhaobo/runs/synth-single3.console.log
```

通过后将设备列表改为 0..7，`--ngpus` 改成 8，输出目录改为 `synth-eight3`，再跑 3 步。每卡 batch 保持 2。合成数据只证明训练链路，不用它评价驾驶能力。

## 8. 在共享盘准备真实 mini 数据与 train/val

这里要区分三类路径：`downloads/openscene` 保存原始归档，`data/raw-camera` 下的 RGB 目录传给 `--data_root`，`data/prepared/train.pkl` 传给 `--data_path`。训练入口既不解压归档，也不把原始日志转换成训练 pickle。

已验收的 `data/prepared` 可以和同一套 RGB 直接复用：核对 `data_manifest.json`、train/val、归一化、token 清单及图片可读性后，设置下方的 `RGB_ROOT` 并跳到第 9 节。要换数据版本或重新划分时，生成到新的目录并在训练、推理中统一改路径；保留已有产物。pickle 的图片键是相对标识，实际图片由 `RGB_ROOT` 解析。

```bash
# [容器]
test ! -e /work/zhaobo/data/prepared || { echo '已有 prepared，请检查并复用，或另选目录'; exit 1; }
export OPENSCENE_REV=a76f840b65e972bc45e56c2adced897498e9a026
hf download OpenDriveLab/OpenScene openscene-v1.1/openscene_metadata_mini.tgz \
  --repo-type dataset --revision "$OPENSCENE_REV" --max-workers 1 \
  --local-dir /work/zhaobo/downloads/openscene

hf download OpenDriveLab/OpenScene \
  openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_0.tgz \
  --repo-type dataset --revision "$OPENSCENE_REV" --max-workers 1 --local-dir /work/zhaobo/downloads/openscene

tar -tzf /work/zhaobo/downloads/openscene/openscene-v1.1/openscene_metadata_mini.tgz | head
tar -xzf /work/zhaobo/downloads/openscene/openscene-v1.1/openscene_metadata_mini.tgz \
  -C /work/zhaobo/data/raw-logs
tar -xzf /work/zhaobo/downloads/openscene/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_0.tgz \
  -C /work/zhaobo/data/raw-camera

export RGB_ROOT=/work/zhaobo/data/raw-camera/openscene-v1.1/sensor_blobs/mini
test -d "$RGB_ROOT" || exit 1
ZB_PREPARED_TMP=$(mktemp -d /work/zhaobo/data/.prepared.XXXXXX) || exit 1
python /work/zhaobo/bootstrap/support.py prepare \
  --logs /work/zhaobo/data/raw-logs --rgb "$RGB_ROOT" \
  --out "$ZB_PREPARED_TMP" --max-train 2048 --max-val 16 &&
  mv -T -n -- "$ZB_PREPARED_TMP" /work/zhaobo/data/prepared
test ! -e "$ZB_PREPARED_TMP" || { echo "数据未发布，保留临时目录：$ZB_PREPARED_TMP"; exit 1; }
```

归档从固定来源取得，但解包前仍应检查路径，不接受含绝对路径或 `..` 逃逸的归档条目。归档保留，不在初学流程中自动删除。

辅助程序会：

- 递归读取原始 mini 日志，不依赖不确定的解包日志层级名。
- 从全局位姿计算相邻局部 `(dx,dy,dθ)`，保留真正的逐帧 token，避免用 log 名覆盖多个预测文件。
- 要求过去 1 秒和当前 RGB 都存在且可解码，要求足够的未来帧和路线信息。
- 按 log 划分训练和留出集，只用训练集拟合分位数；原始真值另外保留，不把归一化裁剪后的真值误当成物理真值。
- 生成 `train.pkl`、`val.pkl`、`train_tokens.yaml`、`val_tokens.yaml`、`norm_stats.json`、`data_manifest.json`。

如果分片 0 留下不足两个完整 log，会明确报错；再用相同下载命令下载 `_1.tgz`、`_2.tgz` 并解包到同一个相机目录，重新生成。不要通过复制相同样本制造独立验证集。

该划分用于学习闭环，**不是官方 navtest，也不用于宣称论文成绩**。从零自建这个小数据集，能避开当前原始 mini 生成器中“scene token 与 log 名混用”等风险。

## 9. 真实数据八卡训练 50 步，并保存模型

```bash
# [容器，已有空闲八卡]
source /work/zhaobo/bootstrap/enter-env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -o pipefail
bash scripts/scripts_train/train_qwen_vla_navsim.sh \
  --model_name_or_path /work/zhaobo/models/Qwen2.5-VL-7B-Instruct \
  --sd_model_path /work/zhaobo/models/stable-diffusion-v1-5 \
  --data_path /work/zhaobo/data/prepared/train.pkl --data_root "$RGB_ROOT" \
  --action_tokenizer_path "$PWD/configs/fast" \
  --device npu --attn_type sdpa --fp bf16 \
  --ngpus 8 --batch_size 2 --num_workers 0 --pin_memory false \
  --deepspeed_config "$PWD/scripts/sft/zero3_offload.json" \
  --max_pixels 150528 --min_pixels 12544 --use_previous_actions false \
  --max_steps 50 --warmup_steps 5 --logging_steps 1 \
  --save_model true --save_steps 25 --save_total_limit 1 \
  --report_to tensorboard --master_port 29555 \
  --output_dir /work/zhaobo/runs/real50 --exp_name navsim \
  2>&1 | tee /work/zhaobo/runs/real50.console.log
```

每次新实验换输出目录；同目录出现 `checkpoint-*` 时入口会自动恢复训练。`RGB_ROOT` 在新 shell 中需重新 export 为第 8 节给出的固定路径。

验收 `/work/zhaobo/runs/real50/navsim`：实际 `global_step=50`，有 safetensors 权重/索引、配置、tokenizer、processor 和 Trainer 状态。只有一行“saved”或只有 tokenizer 文件不算权重保存成功。

第一次保存与重新加载成功前，不增大 batch、不加融合算子、不上多机。原始 checkpoint、自己训练的 checkpoint 和发布的驾驶 checkpoint 分开命名。

## 10. A5 动作推理：提供明确适配，不直接运行 CUDA 脚本

固定代码的 Qwen 推理入口仍硬编码 CUDA/NCCL；另外在关闭历史动作时会把 None 动作段传入 processor，GT 也未截取到 8 步。本手册的辅助工具生成**独立的 NPU 推理副本**，原始 Git 工作树不改动：

```bash
# [容器]
python /work/zhaobo/bootstrap/support.py patch-inference \
  --repo /work/zhaobo/code/DriveVLA-W0 \
  --out /work/zhaobo/tools/inference_qwen_npu.py

DEVICE=cpu python /work/zhaobo/tools/inference_qwen_npu.py --help
```

工具会严格检查原始代码片段，源码版本不同就报错而不是盲改；改动包括 HCCL、LOCAL_RANK 设备绑定、SDPA、图像尺寸参数、早一秒已观测动作段、独立 token 映射、8 步原始 GT、解码完成检查，并记录源码和副本 SHA256。

```bash
mkdir -p /work/zhaobo/results/real50/val-json
export ASCEND_RT_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29555 \
  /work/zhaobo/tools/inference_qwen_npu.py \
  --qwen_hub /work/zhaobo/runs/real50/navsim \
  --output_dir /work/zhaobo/results/real50/val-json \
  --test_data_pkl /work/zhaobo/data/prepared/val.pkl \
  --action_tokenizer_path "$PWD/configs/fast" \
  --token_yaml /work/zhaobo/data/prepared/val_tokens.yaml \
  --norm_stats_path /work/zhaobo/data/prepared/norm_stats.json \
  --raw_img_root "$RGB_ROOT" \
  --cur_frame_idx 3 --future_nums 8 --action_dim 3 \
  --max_pixels 150528 --min_pixels 12544 --bf16 --save_gt
```

不要加 `--use_previous_actions`，保持与训练一致。前一秒到当前的动作是已观测条件；当前待预测的未来动作不得进入生成输入。

输出应有与 val 样本数相同的 JSON，每个 `action` 和 `action_gt_denorm` 都是 8×3、物理单位且数值有限。加载日志中 ROSS 辅助头未用于动作生成的键需要解释；骨干和动作 token embedding 的缺失不能一概忽略。

**适配副本本次只做了语法与变换检查，尚未在新镜像里实际生成动作。** 若该阶段失败，保留日志在本阶段修复，不能跨过它声称闭环完成。

## 11. 最后查看结果：先有图和严格误差

```bash
python /work/zhaobo/bootstrap/support.py report \
  --json-dir /work/zhaobo/results/real50/val-json \
  --state /work/zhaobo/runs/real50/navsim/trainer_state.json \
  --out /work/zhaobo/results/real50/report
```

产物：

```text
results/real50/report/
├── training_loss.png           总 loss 曲线
├── <scene-token>.png           最多8个预测/真值轨迹对比
└── trajectory_metrics.json     场景数、逐场景和平均 ADE/FDE（米）
```

该工具缺少 GT 会报错，不拿预测自身冒充真值；GT 使用数据准备时保留的原始运动。它将相邻运动逐步旋转累积后比较位置，避免把 dx/dy 直接当成未来位置。

查看图片可从自己的电脑只下载结果目录，例如 `scp -r ott9-a55:/data1/zhaobo/drivevla-w0-a55/results/real50/report ./zhaobo-a55-report`。这是取回生成结果，不是覆盖共享代码。

### 11.1 归档验收通过的训练权重

训练和推理验收通过后，把最终推理模型归档到共享盘。下面只复制最终模型目录的顶层文件，`checkpoint-*` 和其中的优化器状态仍留在本机。将来换机可把 `--qwen_hub` 指向归档目录。

```bash
# [容器；仅在确认推理验收完成后执行]
(
  set -euo pipefail
  SRC=/work/zhaobo/runs/real50/navsim
  test -s "$SRC/config.json"
  compgen -G "$SRC/*.safetensors" >/dev/null
  PARENT=/work/zhaobo/models/trained
  DEST="$PARENT/a55-real50-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$PARENT"
  STAGE=$(mktemp -d "$PARENT/.archive.XXXXXX")
  find "$SRC" -maxdepth 1 -type f -exec cp -t "$STAGE" -- {} +
  (
    cd "$STAGE"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -exec sha256sum {} + > SHA256SUMS
    sha256sum -c SHA256SUMS
  )
  mv -T -n -- "$STAGE" "$DEST"
  test ! -e "$STAGE"
  printf 'Archived model: %s\n' "$DEST"
)
```

学会读三个结果：loss 是否有限、预测轨迹和真值怎样偏离、ADE/FDE 的样本数与单位。50 步和少量 mini 场景只是学习验证，不据此评价正式泛化或与论文排名比较。

## 12. PDMS 扩展：评估同一批 mini 留出场景

完成训练、保存、NPU 推理和第 11 节后再做。它需要地图和独立评估依赖，不需要重新训练 Qwen。

### 12.1 地图下载

采用固定仓库 v1.1 下载脚本中的地图来源：

```bash
mkdir -p /work/zhaobo/data/maps
curl --fail --location --retry 3 \
  https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip \
  -o /work/zhaobo/downloads/nuplan-maps-v1.1.zip
unzip -n /work/zhaobo/downloads/nuplan-maps-v1.1.zip -d /work/zhaobo/data/maps
find /work/zhaobo/data/maps -name nuplan-maps-v1.0.json
```

ZIP 文件名是 v1.1，内含地图集合和 NAVSIM 使用的地图版本名可能仍为 `nuplan-maps-v1.0`。应让 `NUPLAN_MAPS_ROOT` 指向包含该地图 JSON 的目录，而不是猜一个上级路径。

### 12.2 独立评估 Python

固定代码里的 NAVSIM 是 1.1.0，原环境选择 Python3.9、numpy1.23.4、torch2.0.1。不要在训练 venv 里直接安装这份 requirements，导致 Qwen 框架被替换。

在自己的容器内安装独立 Python3.9（例如通过独立工具环境中的 uv 管理），创建 `/work/zhaobo/envs/navsim`，再安装固定代码的 `inference/navsim/navsim/requirements.txt` 与 editable NAVSIM。这里老版本 ARM geospatial/PyTorch 依赖仍需在执行时验收；**本次未在干净 A55 上构建这套评估环境，不冒充已测安装清单**。

可执行的环境准备入口为：

```bash
# [容器] 使用单独的工具环境，不修改训练 venv
python3 -m venv /work/zhaobo/envs/tools
/work/zhaobo/envs/tools/bin/python -m pip install uv==0.9.10
/work/zhaobo/envs/tools/bin/uv python install 3.9
/work/zhaobo/envs/tools/bin/uv venv --python 3.9 /work/zhaobo/envs/navsim
/work/zhaobo/envs/navsim/bin/python -m ensurepip
/work/zhaobo/envs/navsim/bin/python -m pip install -e /work/zhaobo/code/DriveVLA-W0/inference/navsim/navsim
```

若原 nuplan-devkit 的 gitee 依赖或特定 ARM wheel 不可用，停止在这一安装节点，保存日志并按相同 nuPlan 版本解决来源/编译依赖；不能默默换成最新版 NAVSIM 或删掉失败样本后报分。必要时另建 CPU 评估镜像，不影响训练镜像与已有结果。

### 12.3 自定义留出场景与缓存

不是 `navtest`，使用 mini split，并把每个 val token、frame_interval 和路径明确传给 Hydra，避免默认非重叠采样漏掉自建场景。

```bash
# [独立 NAVSIM 环境]
export NAVSIM_DEVKIT_ROOT=/work/zhaobo/code/DriveVLA-W0/inference/navsim/navsim
export NAVSIM_EXP_ROOT=/work/zhaobo/results/navsim
export OPENSCENE_DATA_ROOT=/work/zhaobo/data/navsim-layout
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="$(find /work/zhaobo/data/maps -name nuplan-maps-v1.0.json -printf '%h\n' | head -n 1)"
test -n "$NUPLAN_MAPS_ROOT"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

缓存和评分所需的公共 override：

- `train_test_split=mini`；`train_test_split.scene_filter.frame_interval=1`。
- `train_test_split.scene_filter.tokens` 为 `val_tokens.yaml` 中完整 token 列表。
- `navsim_log_path` 为解包出的实际 mini 原始 `.pkl` 日志目录，不能直接指 metadata 的 `.tgz`。
- `sensor_blobs_path` 为 `RGB_ROOT`；评估 JSON agent 不要求 LiDAR 输入，但地图和日志注释必须齐全。
- 缓存用 `cache.cache_path=/work/zhaobo/cache/mini-val`，评分用相同位置的 `metric_cache_path`。

以下把原始日志组织成一个独立平面目录，并执行集合检查、缓存和评分：

```bash
export EVAL_PY=/work/zhaobo/envs/navsim/bin/python
export RGB_ROOT=/work/zhaobo/data/raw-camera/openscene-v1.1/sensor_blobs/mini
export LOG_ROOT=/work/zhaobo/data/navsim-logs/mini
"$EVAL_PY" - <<'PY'
import os
from pathlib import Path
out=Path(os.environ['LOG_ROOT']); out.mkdir(parents=True,exist_ok=True)
for src in Path('/work/zhaobo/data/raw-logs').rglob('*.pkl'):
    dst=out/src.name
    if dst.is_symlink() and dst.resolve()==src.resolve():
        continue
    if dst.exists() or dst.is_symlink():
        raise RuntimeError(f'Duplicate log filename: {dst}')
    dst.symlink_to(src.resolve())
PY

export TOKENS="$("$EVAL_PY" -c 'import json,yaml; print(json.dumps(yaml.safe_load(open("/work/zhaobo/data/prepared/val_tokens.yaml"))["tokens"] ,separators=(",",":")))')"

"$EVAL_PY" - <<'PY'
import os,json
from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter,SensorConfig
tokens=json.loads(os.environ['TOKENS'])
sf=SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,tokens=tokens)
loader=SceneLoader(Path(os.environ['LOG_ROOT']),Path(os.environ['RGB_ROOT']),sf,SensorConfig.build_no_sensors())
pred={p.stem for p in Path('/work/zhaobo/results/real50/val-json').glob('*.json') if not p.stem.endswith('_diff')}
assert set(loader.tokens)==set(tokens)==pred, 'scene / token list / prediction mismatch'
print('Scene and prediction sets match:',len(tokens))
PY

COMMON=(train_test_split=mini worker=sequential \
  train_test_split.scene_filter.frame_interval=1 \
  "train_test_split.scene_filter.tokens=$TOKENS" \
  "navsim_log_path=$LOG_ROOT")

"$EVAL_PY" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  "${COMMON[@]}" cache.cache_path=/work/zhaobo/cache/mini-val

"$EVAL_PY" - <<'PY'
import os,json
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
cache=MetricCacheLoader(Path('/work/zhaobo/cache/mini-val'))
assert set(cache.tokens)==set(json.loads(os.environ['TOKENS'])), 'incomplete metric cache'
print('All validation scenes have a metric cache')
PY

"$EVAL_PY" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  "${COMMON[@]}" "sensor_blobs_path=$RGB_ROOT" \
  agent=emu_vla_agent agent.anchor_based=false \
  agent.experiment_path=/work/zhaobo/results/real50/val-json \
  metric_cache_path=/work/zhaobo/cache/mini-val \
  experiment_name=zhaobo-a55-mini-val
```

这段评分命令在空环境尚未执行；因此场景/缓存集合断言是执行时必须通过的验收门，不是本次已通过的结论。第二次评分前要留意 agent 会在预测目录生成差异 JSON 和图片，严格区分它们与动作预测 JSON。

初学闭环以第 11 节为明确的可视结果终点；PDMS 这一节保留了真实的环境兼容验收点。完整 navtest 则另行下载 test 日志与对应相机分片、使用官方划分重新准备测试输入，不能把 mini 留出集结果叫作 navtest 成绩。

## 13. 执行记录、检查和恢复

| 节点 | 必查结果 |
|---|---|
| 镜像 | ARM64 digest 匹配，构建日志无错误，image.json 已保存 |
| 代码 | remote URL 正确，HEAD 精确等于 2a87094… |
| 环境 | python/torchrun 路径正确，pip check 与训练 --help 通过 |
| 模型 | 分片完整、有 SHA256，原始模型与自己 checkpoint 分开 |
| NPU | 小运算 14.0；八卡通信每 rank 36 |
| 数据 | 至少两个独立 log，train/val 非空、token 无重复、两帧图片可解码 |
| 训练 | 3步/50步真的完成，loss 有限，真实训练模型确实落盘 |
| 推理 | 加载自己的 real50，JSON 数量等于 val，8×3，真值独立 |
| 查看 | loss PNG、轨迹 PNG 和 ADE/FDE JSON 均产生 |
| PDMS | 环境与地图成功、三个 token 集合相等后才允许报分 |

不要运行 `docker rm`、`pkill -f train` 或删除工作区来“重新开始”。遇到失败换新的 runs 子目录，保留日志；仅停止自己确认启动的具体进程。

### 可选：归档验收通过的容器镜像

训练环境验收通过后，可以在 A55 宿主机把**自己的镜像**保存到 `$ZB_SHARED/artifacts/images`，为换机复用。目标机器仍需 ARM64、兼容的宿主驱动和设备透传；镜像 tar 不包含 bind mount 的模型、数据或运行结果。

```bash
# [宿主机，已设置第 2 节的 ZB_SHARED、ZB_IMAGE]
(
  set -euo pipefail
  cd "$ZB_SHARED/artifacts/images"
  ARCHIVE="drivevla-w0-arm64-$(date -u +%Y%m%dT%H%M%SZ).tar"
  STAGE=$(mktemp .image.XXXXXX)
  docker save -o "$STAGE" "$ZB_IMAGE"
  mv -T -n -- "$STAGE" "$ARCHIVE"
  test ! -e "$STAGE"
  sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
)
```

换机器时进入相同共享目录，先 `sha256sum -c <归档文件>.sha256`，再 `docker load -i <归档文件>`，随后按第 3.3 节挂载资源。只归档已经验收的镜像；首次执行仍走本手册的构建步骤。

### 本次实际验证范围

此前 A57 参考调查已核实 Git 分支与提交、模型／数据 revision、torch_npu wheel 元数据和 SHA256。迁到 A55 保留这些资源身份，不复用 A57 的本地文件。

11 月 9 日在 A55 已做：SSH、主机架构、NPU 健康与占用、挂载与磁盘、Docker、设备节点、端口及 SWR 固定镜像 manifest 检查；验证了部分依赖索引和 HF 镜像站的小文件请求，并记录 GitHub 直连超时。

已核实 A55 上的 `/mnt/sfs_turbo/zhaobo` 为可写的共享 NFS 路径，并创建了本项目的资源目录。当前构建和下载状态以文首查询命令为准。

镜像构建、权重下载和数据生成已发起，完成与否以各任务的 `READY` 为准。训练、NPU 动作生成和 PDMS 尚未执行。

## 附录：引导文件

把交付目录中的 Dockerfile 与 support.py 保存到第 3.1 节指定的自己的 bootstrap 目录即可。为避免辅助文件没有被推送到远端仓库而阻塞初学者，下面附有完整文件文本；可在 A55 宿主机用编辑器保存，或使用 `cat > 文件路径 <<'EOF'` 的方式创建。不要保存到参考运行目录。

## 附录 A：创建 Dockerfile（A55 宿主机）

```bash
cat > "$ZB_HOME/bootstrap/Dockerfile" <<'ZHAOBO_DOCKER'
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann@sha256:f81ad0b1f1e9b57e7859a334a9b63ba4444f6af1f0721e87300922627756ec1c
SHELL ["/bin/bash", "-c"]
RUN dnf install -y git gcc gcc-c++ make cmake wget curl tar unzip hostname findutils procps-ng patch diffutils \
    && dnf clean all
RUN python3 -m venv /opt/zhaobo/train
ENV PATH="/opt/zhaobo/train/bin:${PATH}"
ARG ZB_PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL="${ZB_PIP_INDEX_URL}" PIP_DEFAULT_TIMEOUT=90 PIP_RETRIES=3 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m pip install 'setuptools<82' wheel ninja \
    && python -m pip install \
       'torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=70ecb2659af6373b7c5336e692e665605b0201ea21ff51aaea47e1d75ea6b5aa' \
       'torchvision @ https://download-r2.pytorch.org/whl/cpu/torchvision-0.26.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=17f0b542331fc94230b4214c6d123f038af7330fd81019608c0d2402f3bc3079' \
       'torchaudio @ https://download-r2.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=b9dd2c6ac144001dc6dac38b564c1de73ac26ef0c195d5037c4a94990b0e2b5a'
RUN curl --fail --location --connect-timeout 20 --retry 3 --retry-all-errors --output /tmp/torch_npu.whl.download \
       https://files.pythonhosted.org/packages/8b/6d/92268dbc3290c0efd9f4d656c3ed61c7340c6a0c609c156b613347730001/torch_npu-2.11.0.post3-cp312-cp312-manylinux_2_28_aarch64.whl \
    && echo '7d1284333a63671da1df222a1185f3997eafcaa533fe3d9e4aacad556eb27e04  /tmp/torch_npu.whl.download' | sha256sum -c - \
    && mv /tmp/torch_npu.whl.download /tmp/torch_npu-2.11.0.post3-cp312-cp312-manylinux_2_28_aarch64.whl \
    && python -m pip install /tmp/torch_npu-2.11.0.post3-cp312-cp312-manylinux_2_28_aarch64.whl
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && DS_BUILD_OPS=0 python -m pip install deepspeed==0.19.7 \
       accelerate==1.10.1 numpy==1.26.4 scipy==1.13.1 pillow==10.4.0 \
       pyyaml==6.0.2 tqdm==4.67.1 "safetensors>=0.8.0" sentencepiece==0.2.1 \
       tiktoken==0.6.0 einops==0.8.1 diffusers==0.39.0 huggingface_hub==0.36.2 \
       pyquaternion==0.9.9 matplotlib==3.9.4 tensorboard==2.20.0
ENV HF_ENDPOINT="https://hf-mirror.com"
ENV HF_HOME="/work/zhaobo/cache/huggingface"
ENV TORCH_EXTENSIONS_DIR="/work/zhaobo/cache/torch-extensions"
ENV MPLCONFIGDIR="/work/zhaobo/cache/matplotlib"
ENV XDG_CACHE_HOME="/work/zhaobo/cache/xdg"
WORKDIR /work/zhaobo
CMD ["bash", "--noprofile", "--norc"]
ZHAOBO_DOCKER
```

## 附录 B：创建辅助程序（A55 宿主机）

```bash
cat > "$ZB_HOME/bootstrap/support.py" <<'ZHAOBO_PYTHON'
"""Standalone helpers for zhaobo's A55 tutorial; no training is started here."""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(args):
    import numpy as np
    import yaml
    from PIL import Image
    from pyquaternion import Quaternion

    logs, rgb, out = Path(args.logs), Path(args.rgb), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    commands = ["go left", "go straight", "go right", "unknown"]
    for p in sorted(logs.rglob("*.pkl")):
        with p.open("rb") as f:
            frames = pickle.load(f)
        if not isinstance(frames, list) or not frames or not isinstance(frames[0], dict):
            continue
        if "ego2global_translation" not in frames[0]:
            continue
        poses, names, texts = [], [], []
        for f in frames:
            t = f["ego2global_translation"]
            yaw = Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0]
            poses.append([t[0], t[1], yaw])
            parts = str(f["cams"]["CAM_F0"]["data_path"]).split("/")
            k = parts.index("CAM_F0")
            names.append(f"{parts[k-1]}/{Path(parts[-1]).stem}.npy")
            cmd = np.flatnonzero(np.asarray(f["driving_command"]))
            texts.append(commands[int(cmd[0])] if len(cmd) and int(cmd[0]) < 3 else "unknown")
        poses = np.asarray(poses, dtype=np.float64)
        motion = np.zeros_like(poses)
        for i in range(len(poses)-1):
            dx, dy, dyaw = poses[i+1] - poses[i]
            c, s = math.cos(poses[i, 2]), math.sin(poses[i, 2])
            motion[i] = [c*dx+s*dy, -s*dx+c*dy, (dyaw+math.pi)%(2*math.pi)-math.pi]
        samples = []
        for i in range(5, len(frames)-10):
            if len(frames[i].get("roadblock_ids", [])) == 0:
                continue
            cur, prev = list(range(i-3, i+9)), list(range(i-5, i+7))
            paths = []
            for j in (i, i-2):
                name = Path(names[j])
                paths.append(rgb/name.parent/"CAM_F0"/(name.stem+".jpg"))
            if not all(q.is_file() for q in paths):
                continue
            # Decode both input files now, rather than trusting existence alone.
            for q in paths:
                with Image.open(q) as im:
                    im.verify()
            sample = {"token": str(frames[i]["token"]), "log_name": p.stem,
                      "text": [texts[j] for j in cur], "image": [names[j] for j in cur],
                      "action": motion[cur].copy(),
                      "action_raw": motion[cur].copy(),
                      "pre_1s_text": [texts[j] for j in prev],
                      "pre_1s_image": [names[j] for j in prev],
                      "pre_1s_action": motion[prev].copy()}
            if np.isfinite(sample["action"]).all() and np.isfinite(sample["pre_1s_action"]).all():
                samples.append(sample)
        if samples:
            if p.stem in groups:
                raise ValueError(f"Duplicate raw log name: {p.stem}")
            groups[p.stem] = samples
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need >=2 image-complete logs for an independent holdout. Download more camera shards.")
    cut = min(len(keys)-1, max(1, int(len(keys)*0.8)))
    train_keys, val_keys = keys[:cut], keys[cut:]
    train = [s for k in train_keys for s in groups[k]][:args.max_train]
    val = [s for k in val_keys for s in groups[k]][:args.max_val]
    if not train or not val:
        raise ValueError("Empty train/validation split")
    all_tokens = [s["token"] for s in train+val]
    if len(set(all_tokens)) != len(all_tokens):
        raise ValueError("Duplicate frame tokens; cannot name predictions uniquely")
    a = np.concatenate([s["action"] for s in train])
    low, high = np.percentile(a, [1, 99], axis=0)
    if np.any(high-low < 1e-8):
        raise ValueError("Degenerate action quantiles; expand the dataset")
    stats = {"q01": low.tolist(), "q99": high.tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist()}
    for name, data in [("train", train), ("val", val)]:
        for s in data:
            for field in ("action", "pre_1s_action"):
                s[field] = np.clip(2*(s[field]-low)/(high-low+1e-8)-1, -1, 1).astype(np.float32)
        with (out/f"{name}.pkl").open("wb") as f:
            pickle.dump(data, f)
        (out/f"{name}_tokens.yaml").write_text(yaml.safe_dump({"tokens": [s["token"] for s in data]}), encoding="utf-8")
    write_json(out/"norm_stats.json", {"norm_stats": {"libero": stats}})
    write_json(out/"data_manifest.json", {"train_logs": train_keys, "val_logs": val_keys,
        "train_samples": len(train), "val_samples": len(val), "source_logs": str(logs),
        "rgb_root": str(rgb), "horizon": 8, "interval_seconds": 0.5,
        "warning": "Learning subset; no official navtest or paper-score claim"})
    print(f"Prepared {len(train)} train / {len(val)} validation scenes in {out}")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned-source check failed for: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch(args):
    src = Path(args.repo)/"inference/qwen/inference_action_navsim_vava.py"
    text = src.read_text(encoding="utf-8")
    text = replace_once(text, 'PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)',
                        f'PROJECT_ROOT = {str(Path(args.repo).resolve())!r}')
    text = replace_once(text, "import torch\n", "import torch\nimport torch_npu\n")
    text = replace_once(text, "backend='nccl'", "backend='hccl'")
    text = replace_once(text, 'device = torch.device(f"cuda:{rank}")\n    torch.cuda.set_device(device)',
        'local_rank = int(os.environ.get("LOCAL_RANK", 0))\n    device = torch.device(f"npu:{local_rank}")\n    torch.npu.set_device(device)')
    text = replace_once(text, 'attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')
    text = replace_once(text, 'image_processor = AutoProcessor.from_pretrained(args.qwen_hub).image_processor',
        'image_processor = AutoProcessor.from_pretrained(args.qwen_hub).image_processor\n    image_processor.max_pixels = args.max_pixels\n    image_processor.min_pixels = args.min_pixels')
    begin = text.index('        # User historical actions ->')
    end = text.index('        # Build user content with', begin)
    text = text[:begin] + '''        # Training uses no user historical-action segments. The earlier
        # assistant segment contains only motion already observed by time t.
        if args.use_previous_actions:
            raise ValueError("This tutorial matches use_previous_actions=false training")
        pre = np.asarray(scene["pre_1s_action"])[args.cur_frame_idx:args.cur_frame_idx+2]
        pre_ids = action_tokenizer(pre)[0]
        pre_wrapped_action_token_ids = torch.tensor(
            [boa_token_id] + [last_vocab_idx-i for i in pre_ids] + [eoa_token_id], dtype=torch.long)
        if image_tensor is None or pre_image_tensor is None:
            raise ValueError(f"Missing RGB input for scene {scene.get('token', idx)}")

''' + text[end:]
    if 'import numpy as np\n' not in text:
        text = replace_once(text, 'import pickle\n', 'import pickle\nimport numpy as np\n')
    text = replace_once(text, 'action_segments = [pre_user_action_token_ids, pre_wrapped_action_token_ids, user_action_token_ids]',
        'action_segments = [pre_wrapped_action_token_ids]')
    text = replace_once(text, 'action_roles = ["user", "assistant", "user"]', 'action_roles = ["assistant"]')
    text = replace_once(text, 'gt = scene["action"]', 'gt = np.asarray(scene.get("action_raw", scene["action"]))[args.cur_frame_idx:args.cur_frame_idx+args.future_nums]')
    text = replace_once(text, 'gt_denorm = 0.5 * (gt_tensor + 1.0) * (action_high - action_low) + action_low',
        'gt_denorm = gt_tensor if "action_raw" in scene else 0.5 * (gt_tensor + 1.0) * (action_high - action_low) + action_low')
    text = replace_once(text, 'token_name = token_list[idx]',
        'token_name = str(scene["token"])\n        if token_list[idx] != token_name:\n            raise ValueError("token YAML and pickle ordering differ")')
    text = replace_once(text, 'action_tok_ids = [last_vocab_idx - t for t in filtered]',
        'if eoa_token_id not in gen_tokens.tolist() or not filtered:\n            raise ValueError("Incomplete generated action sequence")\n        action_tok_ids = [last_vocab_idx - t for t in filtered]')
    text = replace_once(text, 'pred = torch.tensor(decoded_actions, dtype=torch.float32, device=device)',
        'if np.asarray(decoded_actions).shape != (8, 3) or not np.isfinite(decoded_actions).all():\n            raise ValueError("Invalid decoded actions")\n        pred = torch.tensor(decoded_actions, dtype=torch.float32, device=device)')
    compile(text, str(src), "exec")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}")
    out.write_text(text, encoding="utf-8")
    write_json(out.with_suffix(".provenance.json"), {"source": str(src),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "runtime_status": "Syntax checked only; NPU generation still requires runtime validation"})
    print("Created NPU inference copy:", out)


def report(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, out = Path(args.json_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    def positions(a):
        x = y = yaw = 0.
        result = []
        for dx, dy, dt in a:
            x, y = x+math.cos(yaw)*dx-math.sin(yaw)*dy, y+math.sin(yaw)*dx+math.cos(yaw)*dy
            yaw += dt
            result.append([x, y])
        return np.asarray(result)
    rows = []
    for f in sorted(root.glob("*.json")):
        d = json.loads(f.read_text())
        if "action" not in d:
            continue
        if "action_gt_denorm" not in d:
            raise ValueError(f"Missing ground truth: {f}")
        pred, gt = np.asarray(d["action"]), np.asarray(d["action_gt_denorm"])
        if pred.shape != (8, 3) or gt.shape != (8, 3) or not np.isfinite([pred, gt]).all():
            raise ValueError(f"Bad action array: {f}")
        p, g = positions(pred), positions(gt)
        error = np.linalg.norm(p-g, axis=1)
        rows.append({"token": f.stem, "ADE_m": float(error.mean()), "FDE_m": float(error[-1])})
        if len(rows) <= 8:
            fig, ax = plt.subplots()
            ax.plot(np.r_[0, g[:,0]], np.r_[0, g[:,1]], "o-", label="ground truth")
            ax.plot(np.r_[0, p[:,0]], np.r_[0, p[:,1]], "x--", label="prediction")
            ax.set(xlabel="forward x (m)", ylabel="left y (m)", title=f.stem)
            ax.axis("equal"); ax.legend(); fig.tight_layout()
            fig.savefig(out/f"{f.stem}.png", dpi=150); plt.close(fig)
    if not rows:
        raise ValueError("No predictions to evaluate")
    write_json(out/"trajectory_metrics.json", {"samples": len(rows),
        "mean_ADE_m": float(np.mean([r["ADE_m"] for r in rows])),
        "mean_FDE_m": float(np.mean([r["FDE_m"] for r in rows])), "scenes": rows})
    if args.state:
        d = json.loads(Path(args.state).read_text())
        history = [r for r in d.get("log_history", []) if "loss" in r and "step" in r]
        if history:
            fig, ax = plt.subplots()
            ax.plot([r["step"] for r in history], [r["loss"] for r in history])
            ax.set(xlabel="optimizer step", ylabel="training loss")
            fig.tight_layout(); fig.savefig(out/"training_loss.png", dpi=150); plt.close(fig)
    print("Wrote metrics and plots:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--logs", required=True); p.add_argument("--rgb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-train", type=int, default=2048); p.add_argument("--max-val", type=int, default=64)
    p.set_defaults(func=prepare)
    p = sub.add_parser("patch-inference")
    p.add_argument("--repo", required=True); p.add_argument("--out", required=True); p.set_defaults(func=patch)
    p = sub.add_parser("report")
    p.add_argument("--json-dir", required=True); p.add_argument("--out", required=True)
    p.add_argument("--state"); p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
ZHAOBO_PYTHON
```
