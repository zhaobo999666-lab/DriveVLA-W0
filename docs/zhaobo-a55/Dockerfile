FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann@sha256:f81ad0b1f1e9b57e7859a334a9b63ba4444f6af1f0721e87300922627756ec1c
SHELL ["/bin/bash", "-c"]
RUN dnf install -y git gcc gcc-c++ make cmake wget curl tar unzip hostname findutils procps-ng patch diffutils \
    && dnf clean all
RUN python3 -m venv /opt/zhaobo/train
ENV PATH="/opt/zhaobo/train/bin:${PATH}"
RUN python -m pip install --upgrade pip setuptools wheel ninja \
    && python -m pip install torch==2.11.0+cpu torchvision==0.26.0+cpu torchaudio==2.11.0+cpu \
       --index-url https://download.pytorch.org/whl/cpu
RUN curl --fail --location --retry 3 --output /tmp/torch_npu.whl.download \
       https://files.pythonhosted.org/packages/8b/6d/92268dbc3290c0efd9f4d656c3ed61c7340c6a0c609c156b613347730001/torch_npu-2.11.0.post3-cp312-cp312-manylinux_2_28_aarch64.whl \
    && echo '7d1284333a63671da1df222a1185f3997eafcaa533fe3d9e4aacad556eb27e04  /tmp/torch_npu.whl.download' | sha256sum -c - \
    && mv /tmp/torch_npu.whl.download /tmp/torch_npu-2.11.0.post3-cp312-cp312-manylinux_2_28_aarch64.whl \
    && python -m pip install /tmp/torch_npu-2.11.0.post3-cp312-cp312-manylinux_2_28_aarch64.whl
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && DS_BUILD_OPS=0 python -m pip install deepspeed==0.19.7 \
       accelerate==1.10.1 numpy==1.26.4 scipy==1.13.1 pillow==10.4.0 \
       pyyaml==6.0.2 tqdm==4.67.1 safetensors==0.6.2 sentencepiece==0.2.1 \
       tiktoken==0.6.0 einops==0.8.1 diffusers==0.39.0 huggingface_hub==0.36.2 \
       pyquaternion==0.9.9 matplotlib==3.9.4 tensorboard==2.20.0
ENV HF_ENDPOINT="https://hf-mirror.com"
ENV HF_HOME="/work/zhaobo/cache/huggingface"
ENV TORCH_EXTENSIONS_DIR="/work/zhaobo/cache/torch-extensions"
ENV MPLCONFIGDIR="/work/zhaobo/cache/matplotlib"
ENV XDG_CACHE_HOME="/work/zhaobo/cache/xdg"
WORKDIR /work/zhaobo
CMD ["bash", "--noprofile", "--norc"]
