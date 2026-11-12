# 在 A55（Ascend A5）运行 DriveVLA-W0 的 Qwen2.5-VL 路线

> 当前实施目标是 zhaobo / A55，请使用 [A55 从零实施手册](D:/OTT9/DriveVLA-W0/docs/zhaobo-a55-final-runbook.zh-CN.md)。以下保留为早期复用现成容器的参考。

核查日期：2026-11-06。本说明针对已存在的 `drivevla-w0` 容器，选择 Qwen2.5-VL-7B + ROSS，不是 Emu3 训练环境。

## 1. 当前实机情况

- A55 SSH 别名：`ott9-a55`，8 个 Ascend950DT，单设备 96 GiB HBM。
- 容器：`drivevla-w0:cann9.2.0-pytorch2.11.0-python3.12-a5`。
- 基础 Python：`/opt/cosmos3_py312/bin/python`。
- CANN 9.2.0，torch 2.11.0+cpu，torch_npu 2.11.0.post1.dev20260824，DeepSpeed 0.19.7。这些是实测版本，不是通用版本兼容性保证。
- 基础环境里的 Transformers 4.44.0 用于 Emu3；Qwen 必须另建环境并使用本仓库 Transformers。
- `/mnt/sfs_turbo/lmc/DriveVLA-W0` 是已有项目副本，但不保证和本地工作区完全一致。
- `/mnt/sfs_turbo/lmc/venv-w0` 的解释器链接指向容器内不存在的 Python，不能因为宿主机能看到目录就直接使用。
- 模型目录：`/mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct` 和 `/mnt/sfs_turbo/models/stable-diffusion-v1-5`。
- 本次检查时 8 张 NPU 被其他 Emu3 多机任务占用，日志出现 `HcclAllGather` 错误。本次未停止、重启或加入该任务，也未运行新的训练。

## 2. 将本地代码放到独立远程目录

已有远程项目是共享目录，建议把当前本地工作区上传到独立副本，避免覆盖其他人的代码和未提交文件。在本地 PowerShell 中执行：

```powershell
ssh ott9-a55 'mkdir -p /mnt/sfs_turbo/lmc/DriveVLA-W0-a5-codex'
tar --exclude=.git --exclude=.idea --exclude=.workbuddy --exclude=logs --exclude=__pycache__ -cf - . | ssh ott9-a55 'tar -xf - -C /mnt/sfs_turbo/lmc/DriveVLA-W0-a5-codex'
```

这是示例上传命令，本次没有自动同步整个工程。PowerShell 7.4+ 支持原生程序之间保留二进制管道；老版 PowerShell 请改用 `scp -r` 或其他可靠文件同步工具。

## 3. 进入已有容器

```bash
ssh ott9-a55
docker exec -it -w /mnt/sfs_turbo/lmc/DriveVLA-W0-a5-codex drivevla-w0 bash --noprofile --norc
```

后续命令都在容器内执行。`--noprofile --norc` 避免加载残留的 CANN 9.1 启动路径；脚本会显式加载本机 CANN。无需另建 Docker，也不要在正在运行 Emu3 的基础 Python 中升级 Transformers。

## 4. 准备隔离环境

```bash
bash scripts/a5/qwen.sh setup
bash scripts/a5/qwen.sh check
```

脚本会从容器自带 Python 创建 `/data1/drivevla-w0-codex/qwen-py312`，复用镜像已有的 torch/torch_npu/DeepSpeed，再在虚拟环境内安装本地 Transformers 和 Qwen 依赖。它不会安装 `requirements_qwen.txt` 中的通用 torch_npu wheel 来替换 A5 镜像特定版本。

首次 setup 需要 pip 网络和 C++ 构建工具。已有镜像若缺少 DeepSpeed、torch_npu 或工具链，setup 会报错，需要先补齐与 A5 镜像匹配的组件，而不是无条件替换整个软件栈。

环境通过应看到：

```text
Qwen and ROSS imports: OK (no NPU tensor allocated)
Environment and argument parser checks passed. No training was started.
```

检查还会解析实际训练入口的参数，避免出现模型 import 正常但启动器与参数 dataclass 不匹配的问题。

## 5. 先做单卡、3 步最小训练

先运行 `npu-smi info`，等现有任务释放设备后执行：

```bash
bash scripts/a5/qwen.sh smoke
```

这会生成 64 个合成样本，以单卡、每卡 batch 2、BF16、SDPA、ZeRO-3 CPU offload 训练 3 步，不保存模型。输出目录为 `logs/a5_qwen_smoke_<时间>/navsim`。

这里只验证加载、前向、反向和更新链路，不证明模型具有实际驾驶能力，也不验证八卡 HCCL 通信。完成标志是步数达到 3、训练正常退出，以及 action/ROSS loss 为有限值。

可以覆盖参数做八卡通信验证：

```bash
bash scripts/a5/qwen.sh smoke --ngpus 8
```

默认脚本在发现任何已有 NPU 进程时拒绝启动，以免和现有任务争抢设备；资源检查存在时间窗口，启动前仍应确认设备分配。

## 6. 换真实数据，启动训练

准备 NAVSIM 两时刻 pickle，以及匹配的 RGB 图像目录。不能用 Emu3 `.npy` 视觉码目录代替 RGB 根目录。

```text
DATA_ROOT/
└── <log_name>/
    └── CAM_F0/
        └── <frame>.jpg
```

pickle 至少包含 `token/text/image/action/pre_1s_text/pre_1s_image/pre_1s_action`，动作归一化和图像路径映射应与 Dataset 约定一致。当前发现的 `navsim_trainval_subset.pkl` 等文件不能仅凭名字就认定适合正式训练，需先检查字段、图像可读性和样本来源。

```bash
export DATA_PATH=/实际路径/navsim_trainval.pkl
export DATA_ROOT=/实际路径/sensor_blobs/trainval
bash scripts/a5/qwen.sh train
```

正式模式默认 8 卡、每卡 batch 2、4000 步并保存模型。确认稳定后再增大 batch、增加 DataLoader worker 或切换 attention；初次启动不要直接使用原启动器每卡 batch 64 的默认值。

## 7. 关键约束与后续评估

- Qwen2.5 ROSS 使用固定 10×18 图像 token 网格；脚本采用已配套的 `max_pixels=150528`，输入仍需满足原项目的图像尺寸/比例约定。
- 每卡 batch 不要设为 1：当前 Qwen2.5 ROSS 中的 `squeeze()` 会移除 batch 维度。
- 保持 `use_previous_actions=false`，与当前两动作段的 ROSS mask 约束一致。
- 第一轮使用 `sdpa`，不安装 CUDA 专用的 `flash-attn`。确认基础链路后再验证本地 NPU Flash Attention 路径。
- 训练和模型保存成功后，动作推理脚本还需适配 NPU：目前 `inference/qwen/inference_action_navsim_vava.py` 仍偏 CUDA/NCCL。此启动器不声称已经完成 NPU 推理或 PDMS 复现。
- 普通动作 JSON 必须反归一化并遵循相对运动协议，才能交给 NAVSIM `EmuVLAAgent` 评分。

## 8. 已验证与未验证

已验证：主机资源、已有容器和共享模型目录；此前在加载 CANN 环境后的小张量 NPU 运算；本次新增脚本的 Bash 语法与控制流检查。

未验证：新隔离环境的实际 pip 安装、3 步模型训练、八卡通信、真实数据训练和 NPU 推理。当前有外部任务占用设备，因此本次交付的是可执行启动脚本与逐步运行说明，不是训练成功报告。
