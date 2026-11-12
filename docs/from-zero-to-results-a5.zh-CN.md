# DriveVLA-W0：A5 单机从零搭建、训练、推理到查看结果的学习规划

> 当前实施目标是 zhaobo / A55：[打开 A55 从零实施手册](D:/OTT9/DriveVLA-W0/docs/zhaobo-a55-final-runbook.zh-CN.md)。本文件保留为概念学习材料；机器、代码分支、镜像和路径以 A55 版为准。

日期：2026-11-06。对象：刚接触服务器、Docker 和模型训练的学习者。

主线：**一台 A5 / Ascend950DT 服务器 → Qwen2.5-VL-7B + ROSS → NAVSIM 数据 → 训练 → 保存权重 → 动作推理 → 轨迹与指标。**

本文是一份分阶段执行规划与操作骨架。本次只做只读经验调查和本地文档整理，不安装软件、不下载大型文件、不占用 NPU 训练，也不修改已有任务。

## 1. 先约定“从零”和“跑通”

### 1.1 从零指什么

假设机器上没有项目、Docker 镜像、Python 环境、模型权重和数据文件。所有资料都要知道从哪里取得、下载到哪里、怎样验证，不直接借用 `/mnt/sfs_turbo/models` 或现成 `drivevla-w0` 环境。

公开的预训练模型仍然是这条学习路线的起点。**从空环境搭建 ≠ 从随机参数预训练 Qwen。** 从头训练数十亿参数的通用模型和完整 NuPlan 世界模型预训练不属于第一次学习闭环；后者作为进阶项目另列。

### 1.2 跑通有不同层次

| 级别 | 实际证明的事情 | 尚不能证明的事情 |
|---|---|---|
| NPU 检查通过 | 框架能在设备上计算 | 模型和数据能训练 |
| 合成数据训练完成 3 步 | 模型加载、前向、反向、优化器链路能工作 | 有真实驾驶能力 |
| 真实小数据训练与保存成功 | 实际数据读取、训练和权重落盘能工作 | 达到论文分数 |
| 保存后的权重能重新加载并生成动作 | 模型可以离线推理 | 驾驶行为安全或准确 |
| 轨迹图和 ADE/FDE 可计算 | 能比较预测与真值 | 通过 NAVSIM 仿真评价 |
| NAVSIM PDMS 可计算 | 仿真、地图、缓存和模型输出协议串通 | 复现论文同条件结果 |

第一次学习的目标是逐级达到最后一行，而不是直接追求论文分数。

## 2. 用通俗的话理解工具

| 名词 | 可以怎样理解 | 在本项目里负责什么 |
|---|---|---|
| 宿主机 | 真正的服务器 | 运行操作系统、驱动、Docker，连接 NPU |
| SSH | 远程终端 | 从自己的电脑登录服务器 |
| 驱动 / 固件 | 系统和硬件沟通的底层组件 | 让系统正确识别 A5 |
| CANN | 昇腾计算工具和运行库 | 执行设备算子，供框架调用 |
| Docker 镜像 | 已打包的软件模板 | 定义 Linux 用户态、Python、CANN 等环境 |
| Docker 容器 | 用模板启动的运行空间 | 在里面安装依赖、运行项目 |
| 挂载 | 把宿主机目录接入容器 | 保留代码、数据和训练输出 |
| venv | 一套独立 Python 包目录 | 隔离 Qwen 与 Emu3 的依赖 |
| torch / torch_npu | 神经网络计算框架及昇腾适配 | 加载模型、计算梯度、更新参数 |
| checkpoint | 模型和训练过程的存档 | 用于推理或恢复训练，两种用途所需内容不同 |
| pickle / pkl | Python 保存的结构化数据文件 | 保存样本、图像路径、动作、命令等，不等于原始图像 |
| batch size | 每个进程每次处理的样本数 | 影响显存需求和训练吞吐 |
| step | 一次优化更新 | 不等于“看过全部数据一次”的 epoch |
| 单机八卡 | 一台服务器同时用八张 NPU | 需要单机 HCCL 通信，不需要跨服务器组网 |

本项目中的“部署”首先指把程序部署到训练机，以及完成离线推理。第一次闭环不需要先搭 vLLM、网页界面或 HTTP 服务。

## 3. 参考了什么真实经验

### 3.1 已核实的参考材料

- 通过 A57 只读连接确认机器、已有 `lmc-w0` tmux 会话和容器；未向这些会话发送命令或中断信号。
- 项目 `npu-port.md` 明确记录 A57 的 Qwen2.5-VL 单机 NPU 移植、真实数据处理和性能工作。
- 从共享日志目录读取以下 `trainer_state.json`。**共享文件能从多台机器访问，因此不能仅凭读取位置把每次实验归属到某台机器。**

| 实验相对路径（均位于参考项目的 logs 下） | 实际完成 / 目标 step | 最后一条 step loss | 能支持的结论 |
|---|---:|---:|---|
| `ross_gate_fixed_20261020/navsim` | 3 / 3 | 14.0139 | 合成数据短训练完成 |
| `smoke_w0_20261021/navsim` | 3 / 3 | 14.0115 | 另一轮短训练完成；配套日志有正常结束记录 |
| `profile_l1_best_es_20261026` | 32 / 32 | 4.0544 | 完成短训练与 profiling 记录 |
| `ab_gateup_20261028` | 50 / 50 | 3.5995 | 完成 50 步训练记录 |

这里的数字只用于证明参考训练达到目标步数。它们不是新实验应该达到的门槛，也不是可相互直接比较的驾驶成绩。

### 3.2 不能照搬的部分

- 参考机器上有本地镜像 `drivevla-w0:cann9.2.0-pytorch2.11.0-python3.12-a5`，检查到没有 registry digest；**不能把这个本地标签当成公众可下载的镜像地址**。
- A55 先前实测环境含开发版 `torch_npu 2.11.0.post1.dev20260824`，这不等于公开源必然能安装到这个包。
- 参考目录存在旧路径、针对性能测试的参数，以及仅复制样本扩大长度的 pickle。复制 20 倍不等于新增 20 倍真实驾驶数据。
- 某日志写着“Model saved”仍需检查实际权重文件；仅有 tokenizer 和 trainer_state 不代表可推理的完整模型。
- 当前 Qwen 推理入口还使用 CUDA/NCCL，不能把训练成功延伸成“A5 上推理和 PDMS 已验证”。

## 4. 整体阶段与验收表

| 阶段 | 你要完成的事 | 必须留下的产物 / 验收结果 |
|---|---|---|
| 0 | 明确机器、账号、磁盘、网络和空闲资源 | `machine.txt`，明确使用哪一台机器 |
| 1 | 安装并验收驱动、Docker、A5 配套运行环境 | 宿主 `npu-smi info` 正常 |
| 2 | 从公开来源取得基础镜像、创建自己的容器 | 镜像来源、digest/ID、自己的容器名 |
| 3 | 克隆项目，创建训练 Python 环境 | 代码 commit、Python 路径、依赖清单 |
| 4 | 验证 NPU 和单机通信 | 小张量计算、八卡集合通信正确 |
| 5 | 下载并验收模型与 FAST | 模型清单、revision、权重分片完整 |
| 6 | 下载少量真实 RGB 和训练 metadata | 图片存在、样本能读取、字段正确 |
| 7 | 合成数据单卡 3 步，再八卡 3 步 | 正常退出、有限 loss、无异常进程残留 |
| 8 | 真实小数据训练 50 步并保存 | 训练日志、checkpoint、tokenizer 和配置 |
| 9 | 重新加载保存的权重 | missing/unexpected keys 可解释，新增动作 token 保留 |
| 10 | 适配并运行 NPU 动作推理 | 每场景 8×3 动作 JSON，数量与场景集合一致 |
| 11 | 画轨迹、算 ADE/FDE | 对比图和严格使用 GT 的误差汇总 |
| 12 | 配 NAVSIM v1.1，下载地图并建立缓存 | 对应 split 的完整缓存 |
| 13 | 计算 PDMS 并整理实验报告 | PDMS CSV、子指标、失败场景和复现清单 |

**每一阶段通过后才进入下一阶段。** 首次不要同时下载全量数据、改 attention、做多机通信和开性能 profiling。

## 5. 从空目录开始的目录规范

宿主机统一使用一个全新目录；以下为示例，不依赖参考机已有目录：

```bash
# [宿主机] 一次性设置；重新登录后需要重新 export
export LAB_HOME=/data1/drivevla-from-zero
mkdir -p "$LAB_HOME"/{code,models,data,downloads,wheels,runs,results,cache,manifests,envs}
```

挂载后容器内统一叫 `/work`：

```text
/work/
├── code/DriveVLA-W0/        项目代码
├── models/
│   ├── Qwen2.5-VL-7B-Instruct/
│   ├── stable-diffusion-v1-5/
│   └── fast/               原始 FAST 下载，项目改版继续使用 configs/fast
├── data/
│   ├── metadata/           下载的训练/测试 pickle
│   └── navsim/
│       ├── sensor_blobs/trainval/
│       ├── sensor_blobs/test/
│       ├── navsim_logs/trainval/
│       ├── navsim_logs/test/
│       └── maps/
├── envs/train/             训练环境
├── envs/navsim/            独立评估环境（或独立评估容器）
├── downloads/              下载归档，初期保留用于校验
├── wheels/                 固定版本安装包及 SHA256
├── runs/                   每次训练一个新目录
├── results/                预测 JSON、图片、指标
├── cache/                  下载缓存、NAVSIM 指标缓存
└── manifests/              机器、环境、代码、数据版本记录
```

容器删除后，挂载目录中的结果仍在宿主机上。未挂载的容器内路径不具有同样的持久性。

## 6. 阶段 0–2：机器、驱动与新容器

### 要学会的事

区分“登录宿主机”和“进入容器”；知道容器不能代替宿主机的硬件驱动。

### 宿主机验收

```bash
# [宿主机]
hostname
uname -m
cat /etc/os-release
npu-smi info
free -h
df -h
docker version
ls -l /dev/davinci*
```

验收：预期是 aarch64，设备类型匹配 Ascend950DT，设备健康，无自己未获分配的训练资源。HBM 已被占用时先协调使用时段，不停止其他人的进程。

如果真是裸机，没有驱动/Docker：先按服务器型号、操作系统和昇腾官方配套文档安装固件、驱动、Docker及设备容器运行组件。具体驱动包名与安装参数依机型和系统而变，**不提供一条猜测版本的万能安装命令**。此步完成的定义是宿主 `npu-smi info` 正常，不是“下载了安装包”。

### 版本选择

可以把 CANN 9.2.0 / PyTorch 2.11.0 / Python 3.12 作为接近经验环境的基线。官方当前配套表列有 CANN 9.2.0 + PyTorch 2.11.0 + torch_npu 2.11.0.post3；它和参考机开发版包不同，仍需在本机逐级验证。[Ascend 官方版本配套](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)

先取得并检查适合 950DT、aarch64、Python 3.12 的 CANN 基础镜像。参考机确实存在以下镜像标签，但下载权限和当前可用性需在执行时验证：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-950-openeuler24.03-py3.12
```

`docker pull` 成功后记录 `docker image inspect`，后续尽量按 digest 固定版本。不要用参考机本地 `drivevla-w0:...` 标签替代基础镜像来源。

### 创建自己的容器：命令骨架

下面的 `A5_BASE_IMAGE` 必须先设置成已取得且确认架构匹配的基础镜像；执行前检查设备映射符合机器实际情况。

```bash
# [宿主机]
export A5_BASE_IMAGE='上一步实际拉取成功的镜像地址'
DEVICES=()
for dev in /dev/davinci{0..7} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [ ! -e "$dev" ] || DEVICES+=(--device "$dev")
done

docker run -dit --name drivevla-learn \
  --network host --shm-size=16g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  "${DEVICES[@]}" \
  -v "$LAB_HOME:/work" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -w /work "$A5_BASE_IMAGE" bash

docker exec -it drivevla-learn bash --noprofile --norc
```

基础镜像需提供 Bash；设备节点或宿主驱动路径不同，应按厂商容器部署文档调整。本骨架不是已在所有 A5 系统上测试过的容器模板。

验收：能进入容器，能看到 `/work`；torch_npu 安装后能识别分配的设备。若宿主能识别而容器不能，先查设备映射、驱动库、环境变量与权限，不重装项目代码。

## 7. 阶段 3：代码和 Python 环境

### 取得代码并固定版本

```bash
# [容器]
cd /work/code
git clone https://github.com/BraveGroup/DriveVLA-W0.git
cd DriveVLA-W0
git rev-parse HEAD
```

公开主仓库和当前工作区带 NPU 修改的版本不保证相同。执行者必须选择一个含 `utils/train_qwen_vla.py`、本地 Transformers、Qwen ROSS 和 NPU 适配的 commit。若要复现当前 fork，应通过其可访问仓库取得对应提交，**不能声称仅 clone 任意最新 main 就包含本文全部修复**。

现有 `scripts/a5/qwen.sh` 是本次对话在本地新增、尚未提交的辅助脚本，干净 clone 不一定有它。本规划以下采用项目主启动器的显式参数，不依赖该未发布脚本。

记录：仓库地址、commit、工作区修改和 patch。代码传播遵循所选项目的工作流：在本地改动、审核后提交，再到机器拉取；不把共享运行目录当作临时编辑器。

### 创建干净训练环境

```bash
# [容器] Python 3.12 和编译工具需先按基础镜像的包管理器安装/确认
python3.12 --version
command -v gcc g++ make git wget tar

python3.12 -m venv /work/envs/train
source /work/envs/train/bin/activate
python -m pip install --upgrade pip
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m pip install \
  torch==2.11.0+cpu torchvision==0.26.0+cpu torchaudio==2.11.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch_npu==2.11.0.post3
```

这是按公开配套表选择的安装基线；需要确认所用包源确实提供 aarch64/Python3.12/A5 对应 wheel。若没有，应取得官方配套包并存入 `/work/wheels`，而不是改成一个看起来版本相近的 CUDA wheel。记录 wheel 名称和 SHA256。

安装项目层依赖：

```bash
cd /work/code/DriveVLA-W0
python -m pip install -e ./reference/transformers
python -m pip install \
  deepspeed==0.19.7 accelerate tensorboard scipy pillow pyyaml tqdm \
  safetensors sentencepiece tiktoken einops==0.8.1 \
  diffusers==0.39.0 huggingface_hub==0.36.2
```

DeepSpeed 0.19.7 是 A55 基础容器检查到的版本；干净安装后的 CPUAdam/HCCL 等需重新验收。编译工具缺失会在 offload 阶段暴露，不能只看 `pip install` 返回成功。

不要把 Emu3 的 `requirements.txt`、旧 Ascend 依赖清单与 Qwen 清单全部叠加安装。Qwen 需要本地修改过的 Transformers；CUDA 专用 `flash-attn` 不属于首次 SDPA 路线的依赖。

检查 `which python`、`which torchrun`，两者应来自 `/work/envs/train/bin`。本步骤在 venv 中安装 torch，通常会同时生成该环境自己的 torchrun；若复用 system-site-packages，需要额外确认 torchrun 的 shebang 没有绕过环境。

## 8. 阶段 4：先验收硬件链路

### 小张量计算

```bash
# [训练环境]
python - <<'PY'
import torch, torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('available:', torch.npu.is_available())
print('devices:', torch.npu.device_count())
x = torch.tensor([1., 2., 3.], device='npu:0')
print('result:', (x*x).sum().cpu().item())
PY
```

验收：设备可见，结果为 `14.0`。不要把 `npu-smi` 的驱动版本显示误当成容器 CANN toolkit 版本。

### 单机八卡通信

准备一个只有设备绑定、`init_process_group('hccl')` 和 `all_reduce` 的短脚本，以 `LOCAL_RANK` 绑定 NPU，八个 rank 分别输入 1..8，求和后每个 rank 都应得到 36，再销毁进程组。

用 `torchrun --standalone --nproc_per_node=8` 启动，设有限超时；先通过这一项，才尝试八卡模型训练。若失败，问题范围是分布式运行环境、设备绑定和通信，不是下载的驾驶样本。

保存验收日志和 `pip freeze`、`python -m pip check` 输出到 `/work/manifests`。

## 9. 阶段 5：从公开来源下载模型

### 需要哪些模型

| 资源 | 用途 | 来源 |
|---|---|---|
| Qwen2.5-VL-7B-Instruct | 视觉语言骨干 | [Qwen 官方模型页](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Stable Diffusion 1.5 | ROSS 的 UNet、VAE、scheduler | [SD1.5 模型页](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| FAST | 动作序列编码参考 | [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) |
| DriveVLA 发布的 metadata | 已整理的驾驶样本入口 | [DriveVLA 模型资源页](https://huggingface.co/liyingyan/DriveVLA-W0) |

**Qwen 原始预训练权重不是你已经训练好的驾驶模型；DriveVLA 发布的驾驶 checkpoint 也不是第一次小训练必须下载的起点。** 可另下载后者作为对照，但应单独记录。

### 下载命令示例

```bash
# [训练环境] hf 命令来自 huggingface_hub
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /work/models/Qwen2.5-VL-7B-Instruct

hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --include 'unet/*' 'vae/*' 'scheduler/*' 'model_index.json' \
  --local-dir /work/models/stable-diffusion-v1-5

hf download physical-intelligence/fast \
  --local-dir /work/models/fast
```

`hf download` 支持下载整个仓库、选取文件以及指定 revision。正式记录应补上 `--revision` 固定提交。[Hugging Face 下载说明](https://huggingface.co/docs/huggingface_hub/guides/cli)

若受限资源需要登录，在终端按官方认证流程完成；不把访问令牌写进教程、脚本或训练日志。网络问题先验证资源地址、账号和下载完成情况，不通过反复切换软件版本解决下载错误。

### 文件验收

- Qwen：`config.json`、tokenizer/processor 配置、权重索引，以及索引引用的**全部** safetensors 分片。
- SD1.5：UNet/VAE 的配置和实际权重、scheduler 配置；只有配置文件不算准备好。
- FAST：模型配置、tokenizer 和 processor 实现。训练优先使用所选代码 commit 的 `configs/fast`，因为它包含项目的解码处理；原始下载留作来源和版本对照。
- 对所有下载生成清单，记录来源、revision、大小、校验值。不要用几个字节的 Git LFS 指针文件代替真实权重。

## 10. 阶段 6：从零下载并整理数据

### 三种数据分别服务什么目标

| 数据级别 | 取得方法 | 第一次用来做什么 |
|---|---|---|
| 合成数据 | 仓库 `synth_qwen_vla_data.py` 本地生成 | 排除下载/路径因素，验证训练 |
| 少量真实 NAVSIM 数据 | 下载部分相机分片 + DriveVLA metadata，再筛选完整样本 | 小规模真实训练和可视化 |
| 正式训练/测试数据 | 按 NAVSIM v1.1 split 下载、检查，并备齐日志与地图 | 有意义的测试和 PDMS |

不先下载整个 OpenScene 数据集。原始数据来源是 [OpenDriveLab/OpenScene](https://huggingface.co/datasets/OpenDriveLab/OpenScene)，NAVSIM v1.1 的安装说明要求传感器、日志和地图分别准备。[NAVSIM v1.1 数据布局](https://raw.githubusercontent.com/autonomousvision/navsim/v1.1/docs/install.md)

### 10.1 metadata

从 DriveVLA 资源页核实文件的真实仓库相对路径后，下载 `navsim_emu_vla_256_144_trainval_pre_1s.pkl` 和后续所需 test pickle 到 `/work/data/metadata`。文件可能位于仓库子目录；下载前先查列表，不凭文件名猜 URL。只下载 metadata 不会同时获得原始 RGB。

虽然名字带 `emu`，Qwen 数据集也能读取其中的动作和索引，并把 `.npy` 路径映射到 RGB；这不表示 Qwen 在读取 Emu3 图像 token。

### 10.2 少量 RGB

```bash
# [训练环境] 只下载第 0 个相机分片
cd /work/code/DriveVLA-W0
bash tools/pickle_gen/download_openscene_camera.sh \
  --out /work/data/navsim-download --start 0 --end 0 \
  --mirror https://huggingface.co
```

按照该脚本的解包约定，图像根目录是：

```text
/work/data/navsim-download/openscene-v1.1/sensor_blobs/trainval
```

此脚本会在解包后删除归档。如需保留归档和校验，应在本地受版本控制的副本中调整这一行为后再部署。不要把“下载命令结束”当作图片可读验收。

### 10.3 过滤为真正能读取的样本

```bash
python tools/pickle_gen/filter_pickle_by_images.py \
  --input /work/data/metadata/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
  --data_root /work/data/navsim-download/openscene-v1.1/sensor_blobs/trainval \
  --cur_frame_idx 3 \
  --output /work/data/metadata/train_subset.pkl
```

过滤后必须有非零样本，而且足够组成几个 batch。一个相机分片能留下多少样本以实际检查为准，不保证等于参考实验的 467。

抽样打开 `image[3]` 和 `pre_1s_image[3]` 对应的两张 RGB，检查前后时间不重复、命令有效、action 维度正确、归一化值有限。边界补帧与不足历史的样本要识别。

train/validation/test 按 log 或场景分组，记录 token 清单，避免重叠时间窗口随机混入不同集合。只复制 pickle 或复用相同图片不能产生独立验证集。

## 11. 阶段 7：先训练 3 步

所有长任务使用**自己新建的** tmux 会话，不使用共享的 `lmc-w0`：

```bash
# [宿主机]
tmux new -s drivevla-learn
docker exec -it drivevla-learn bash --noprofile --norc
```

容器内重新加载环境：

```bash
source /work/envs/train/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/code/DriveVLA-W0
export PYTHONPATH="$PWD:$PWD/reference/Qwen2.5-VL/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tools/synth/synth_qwen_vla_data.py \
  --out_root /work/data/synthetic --num_scenes 64
```

首次单卡训练：

```bash
set -o pipefail
bash scripts/scripts_train/train_qwen_vla_navsim.sh \
  --model_name_or_path /work/models/Qwen2.5-VL-7B-Instruct \
  --sd_model_path /work/models/stable-diffusion-v1-5 \
  --data_path /work/data/synthetic/synth_navsim.pkl \
  --data_root /work/data/synthetic/images \
  --action_tokenizer_path "$PWD/configs/fast" \
  --device npu --attn_type sdpa --fp bf16 \
  --ngpus 1 --batch_size 2 --num_workers 0 --pin_memory false \
  --deepspeed_config "$PWD/scripts/sft/zero3_offload.json" \
  --max_steps 3 --warmup_steps 0 --logging_steps 1 \
  --max_pixels 150528 --min_pixels 12544 \
  --use_previous_actions false --save_model false \
  --report_to none --master_port 29555 \
  --output_dir /work/runs/synth-single3 --exp_name navsim \
  2>&1 | tee /work/runs/synth-single3.console.log
```

通过后换一个新的输出目录，将 `--ngpus` 改为 8，再训练 3 步。全局 batch = 每卡 batch × 卡数 × 梯度累积步数；此处八卡时为 2×8×1=16。

最初固定 SDPA、每卡 batch 2、低 worker 数、关闭 profiling。Qwen2.5 ROSS 中 `squeeze()` 的现状要求每卡至少 2；固定 10×18 token 网格要求配套尺寸参数；`use_previous_actions` 保持 false 以满足两动作段约束。

验收：实际达到 step 3，进程正常退出，loss/grad norm 是有限值，没有 NaN、OOM 或 HCCL 异常。合成数据 loss 不一定下降，不用参考实验的 14.0115 作为通过标准。

## 12. 阶段 8–9：真实数据训练与权重验收

把上一步命令改为：

```text
data_path       /work/data/metadata/train_subset.pkl
data_root       实际下载的 trainval RGB 根目录
ngpus           8
max_steps       50
save_model      true
save_steps      25
save_total_limit 1
output_dir      /work/runs/real50
exp_name        navsim
```

其他参数先保持不变。首次使用真实数据，先不要调整 batch、attention 和融合算子来追吞吐。50 步用于链路验证，不是性能复现；下一轮再根据数据量增加步数。

应该得到：

```text
/work/runs/real50/navsim/
├── config.json
├── model.safetensors 或 model-*.safetensors + 索引
├── tokenizer / processor 相关文件
├── trainer_state.json
├── checkpoint-*/            保存策略和训练方式决定具体内容
└── 日志 / TensorBoard events
```

必须另起 Python 进程重新加载**这个输出目录**，不能又加载原始 Qwen 后误以为训练权重验证成功。检查 BOA/EOA 和动作词表映射、模型配置、权重分片，解释加载时的 missing/unexpected keys。

推理导出只需要模型与 tokenizer 等；精确断点续训还需要相应优化器、scheduler、RNG 和 Trainer 状态。只有最终 safetensors 不等于具备完整续训状态。

如需做一次断点恢复学习实验，保留一个 checkpoint，给同一个实验目录继续增加少量步数，核对日志确实从已有 step 接着训练；不要让自动 resume 意外复用另一实验的目录。

## 13. 阶段 10：把已训练模型用于 A5 推理

**这是当前仓库尚需实施和验证的工程步骤。** 现有 `inference/qwen/inference_action_navsim_vava.py` 硬编码 NCCL、CUDA 和 Flash Attention。不能直接贴一条命令并宣称已经支持 A5。

规划的适配清单：

1. NPU 路线导入 torch_npu，以 `LOCAL_RANK` 绑定设备，使用 HCCL 初始化分布式；保留 CUDA 路线供对照。
2. 将模型和输入搬到正确设备，整数 token/grid/position ID 不误转换成浮点；首轮使用 SDPA。
3. 使用已训练 checkpoint 与保存的 tokenizer/processor，不重新生成另一套动作 token 映射。
4. 推理分辨率、时间窗口、动作 horizon 和上下文格式与训练保持一致。
5. 生成输入在最后一个动作起始标记处正确截断，不把待预测的真值动作作为上下文泄漏进去。
6. 严格用 scene token 命名输出，检查多 rank 场景覆盖、重复、缺失和 YAML/pickle 顺序。
7. FAST 解码后用训练时同一组统计反归一化；任何解码回退/补零单独记录，不静默当作正常样本。
8. 首先生成 1–10 个场景，检查形状为 8×3、数值有限、坐标单位一致，再扩大推理场景集合。

预期单个输出：

```json
{
  "action": [[0.0, 0.0, 0.0]],
  "action_gt_denorm": [[0.0, 0.0, 0.0]]
}
```

上面只展示字段形状，不是可用于评估的真实数值；实际应各有 8 行。`action_gt_denorm` 仅由独立标签读取，不能参与动作生成。

如果需要先学习“推理→评估”而不等自己训练完成，可以额外下载项目发布的 Qwen 驾驶 checkpoint 作对照，记录它与自己训练模型的区别。A5 推理入口的适配依然要完成。

## 14. 阶段 11：先看得见结果，再讨论分数

### 第一次必须交付的图片

- 训练总 loss 随 step 的曲线；若记录了 action_loss / ross_loss，分别画出来。
- 选取几个固定场景，在同一张局部坐标图中叠加真值与预测轨迹，坐标轴单位为米。
- 辅以对应前视图和命令，便于理解左转、直行、右转案例；不把偶然挑选的好案例当作整体效果。

预测是相邻步局部 SE(2) 增量时，应先逐步旋转和累积得到统一坐标中的位姿，不能直接把 dx/dy 的 8 个值当成 8 个位置点。

### ADE/FDE

- ADE：未来各时间步的位置误差平均值。
- FDE：最后一个时间步的位置误差。
- 两者都需要独立真值，单位通常为米；明确报告样本数、horizon、失败数和过滤规则。

当前 `scripts/scripts_infer/eval_action_metrics.py` 在缺少真值时会退回用预测自身比较，可能得到误导的零误差。执行前必须修正为“缺少 GT 就报错或跳过并计数”，同时区分其他评估产生的差异 JSON 和真正的动作 JSON。

小验证集上指标偏低不证明泛化；不要使用训练集自身得出的数字冒充独立测试结果。

## 15. 阶段 12–13：NAVSIM 评估环境、地图、缓存与 PDMS

### 环境分离

训练依赖的本地 Transformers 和 NAVSIM 的 nuPlan/地图/几何依赖可能不同。使用独立 Python 环境或独立评估容器；两者通过 `/work/results` 中的动作 JSON 交换数据。评估环境不必安装 Qwen 权重或 ROSS UNet。

锁定 NAVSIM v1.1 或当前仓库内嵌且包含 `EmuVLAAgent` 的匹配版本，不直接安装另一个最新 NAVSIM 大版本后比较 PDMS。

### 还需从零下载什么

- 与评估 split 一致的 NAVSIM 日志。
- 对应传感器数据，至少满足选定流程和可视化需求。
- nuPlan 地图。
- `navtest` 等正确的场景 token/filter 配置。
- 根据这些资源**生成**的 Metric Cache；它不是模型权重，也不是训练集 pickle。

官方 v1.1 下载说明和仓库 `inference/navsim/navsim/docs/install.md` 给出了原始组织方式。内嵌 `download/` 个别脚本含注释掉的下载或旧环境内容，执行前核对实际脚本，不直接把任意同名脚本当作完整下载器。

### 环境变量与任务骨架

```bash
# [独立 NAVSIM 评估环境]
export NAVSIM_DEVKIT_ROOT=/work/code/DriveVLA-W0/inference/navsim/navsim
export OPENSCENE_DATA_ROOT=/work/data/navsim
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=/work/data/navsim/maps
export NAVSIM_EXP_ROOT=/work/results/navsim
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  cache.cache_path=/work/cache/navtest

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split=navtest agent=emu_vla_agent \
  agent.experiment_path=/work/results/real50/navtest-json \
  agent.anchor_based=false \
  metric_cache_path=/work/cache/navtest \
  experiment_name=drivevla-learn-real50
```

这些是待执行和验证的命令骨架，要求日志、地图、缓存和预测场景完整匹配。若只评估一个测试子集，需要同时限定 scene filter/缓存/JSON 集合；不能把缺失预测简单过滤掉仍称为完整 navtest 成绩。

### 最后怎么看结果

输出逐场景 PDMS 与子指标 CSV，以及预测与真值轨迹图。关注：无责任碰撞、可行驶区域、进度、TTC、舒适性和总分。

总分之外一定检查有效场景数、失败原因和失败样本图。ADE/FDE 反映轨迹误差，PDMS 反映仿真规则下的规划表现；二者没有“数值换算关系”。

## 16. 初学者排错顺序

| 现象 | 第一检查点 | 不宜直接做什么 |
|---|---|---|
| SSH 连不上 | 地址、跳板、端口、密钥和网络 | 重装训练依赖 |
| 宿主看不到 NPU | 驱动、固件、硬件健康 | 改数据集 |
| 容器看不到 NPU | 设备映射、驱动库、CANN source、运行权限 | 认为下载模型错了 |
| `ASCEND_HOME_PATH` 缺失 | 是否在当前 shell source CANN | 重装所有 Python 包 |
| `Qwen2_5_VL...` 无法导入 | Python/Transformers 实际路径、是否装了本地树 | 把 Emu3 requirements 再装一遍 |
| 安装 ARM wheel 失败 | aarch64、Python ABI、发行源和版本组合 | 换一个 CUDA 包 |
| 单卡成功、八卡失败 | LOCAL_RANK 绑定、HCCL、通信测试、资源占用 | 同时更换数据和模型 |
| OOM | 实际空闲 HBM、batch、序列长度、ZeRO/offload、checkpointing | 一开始追大 batch |
| `reshape` 尺寸不对 | Qwen2.5 图像 token 网格和输入尺寸 | 随意截断 hidden states |
| batch=1 时出错 | ROSS `squeeze()` 维度问题 | 认为设备性能不足 |
| 图片找不到 | pkl→RGB 的路径映射、历史帧下载是否完整 | 删除全部 metadata 重来 |
| 进程运行但没有 step | 首次编译/权重加载/数据读取/通信日志 | 只看 NPU 占用就判断卡死 |
| 训练结束却没权重 | save_model/save_strategy、输出位置、磁盘和保存日志 | 仅相信一行“saved”文字 |
| ADE/FDE 全是 0 | GT 是否存在、是否误用预测自身 | 宣布模型完美 |
| PDMS 找不到 cache | split、token、地图版本和 cache 路径 | 重新训练 Qwen |

## 17. 每次实验的记录模板

```text
实验名：
日期 / 使用的机器 / NPU 分配：
代码仓库 / commit / patch：
基础镜像来源 / digest 或 image ID：
驱动 / CANN / Python / torch / torch_npu / Transformers / DeepSpeed：
模型来源 / revision / checkpoint：
数据来源 / split / token 清单 / 唯一样本数：
归一化统计来源：
卡数 / 每卡 batch / 梯度累积 / 全局 batch / 步数：
完整命令 / 环境变量 / 输出目录：
是否保存 / 是否成功重新加载：
推理场景数 / 解码失败数 / 缺失与重复数：
ADE / FDE / PDMS / 有效评估数：
loss 图 / 轨迹图 / 失败案例：
结论和下次只改变的一个因素：
```

最终应能把 `/work/manifests`、对应 runs、预测 JSON、指标 CSV 和图片连起来，回答“这个结果到底由哪份代码、环境、权重和数据产生”。

## 18. 建议的学习节奏与交付包

不承诺固定天数：大文件下载、A5 安装包取得、设备排期、通信与推理适配会影响时间。按验收里程碑推进：

1. **环境包**：宿主/容器/venv 的区别、小张量与八卡通信、版本清单。
2. **资源包**：自己从来源下载的权重、真实 RGB、metadata，完成完整性和路径检查。
3. **训练包**：单卡 3 步、八卡 3 步、真实 50 步、保存与重加载。
4. **结果包**：NPU 推理适配、动作 JSON、loss 和轨迹图、严格 GT 的 ADE/FDE。
5. **评估包**：独立 NAVSIM v1.1 环境、日志/地图/缓存、PDMS 与复现报告。

进阶放在上述闭环之后：全量 NAVSIM、NuPlan 世界模型预训练、Emu3 主线、Qwen3-VL、性能优化、多机训练，以及 HTTP 服务化。它们不是第一次从零部署的必选项。

## 19. 目前明确需要补齐的工程任务

本文没有把已知缺口藏在“直接运行”后面：

- 固定一个能公开取得的 A5 基础镜像与完整 wheel 清单，构建并验收自己的镜像；本地开发版镜像标签不是来源。
- 确定包含所有必要 NPU 修改的代码 commit，并记录未发布改动的传播方式。
- 编写和验收最小 HCCL all_reduce 测试、模型分片及数据一致性检查。
- 完成 Qwen 推理的 NPU/HCCL/SDPA 路径，并验证生成上下文与动作词表。
- 修正 ADE/FDE 缺少真值时的回退行为，输出失败/跳过统计。
- 在独立环境完成 NAVSIM v1.1 缓存与评分，核对全套 token 和物理坐标协议。

这些都是规划中的具体实施项，尚不能声称本次已经从空机器完成全链路复现。
