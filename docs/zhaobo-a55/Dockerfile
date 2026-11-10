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
