# 1) Base system dependencies + Python 3.11
FROM nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04

# 2) assign workdir and set noninteractive mode for apt-get
WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# 3) Base system dependencies + Python 3.11
RUN apt-get update -y \
     && apt-get install -y --no-install-recommends \
     apt-transport-https \
     ca-certificates \
     dbus \
     fontconfig \
     gnupg \
     libasound2 \
     libfreetype6 \
     libglib2.0-0 \
     libnss3 \
     libsqlite3-0 \
     libx11-xcb1 \
     libxcb-glx0 \
     libxcb-xkb1 \
     libxcomposite1 \
     libxcursor1 \
     libxdamage1 \
     libxi6 \
     libxml2 \
     libxrandr2 \
     libxrender1 \
     libxtst6 \
     libgl1-mesa-glx \
     libxkbfile-dev \
     libmagic1 \
     libmagic-dev \
     openssh-client \
     wget \
     xcb \
     xkb-data \
     python3.10 \
     python3.10-venv \
     python3.10-dev \
     python3-pip \
     python3-setuptools \
     git \
     && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
     && ln -sf /usr/bin/pip3 /usr/bin/pip \
     && python -m pip install --upgrade pip \
     && rm -rf /var/lib/apt/lists/*

# 4) Qt6 for Nsight Compute GUI
RUN apt-get update -y \
     && apt-get install -y --no-install-recommends qt6-base-dev \
     && rm -rf /var/lib/apt/lists/*

# 5-9) Python requirements
RUN python -m pip install --upgrade pip \
     && pip install --no-cache-dir setuptools wheel packaging ninja
# Need to first install pytorch then flash attention to avoid the conflict of the version.
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
     && rm /tmp/requirements.txt

# 10) Add NVIDIA repo keyring & install cuDNN + Nsight tools (APT)
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
     && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
     && dpkg -i cuda-keyring_1.0-1_all.deb && rm -f cuda-keyring_1.0-1_all.deb \
     && apt-get update -y \
     && apt-get install -y --no-install-recommends \
     libcudnn9-cuda-12 \
     nsight-compute-2025.3.1 \
     nsight-systems-2025.3.2 \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# 11) Sanity checks: fail the build if tools aren't available
RUN ncu --version && nsys --version && which ncu && which nsys