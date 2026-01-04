FROM nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# 1) Base system dependencies + Python 3.10
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
     && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
     && ln -sf /usr/bin/pip3 /usr/bin/pip \
     && python -m pip install --upgrade pip \
     && rm -rf /var/lib/apt/lists/*

# 2) Qt6 for Nsight Compute GUI
RUN apt-get update -y \
     && apt-get install -y --no-install-recommends qt6-base-dev \
     && rm -rf /var/lib/apt/lists/*

# 3) Python requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
     && rm /tmp/requirements.txt

# 4) Add NVIDIA repo keyring & install cuDNN (with cache purge)
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
     && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
     && dpkg -i cuda-keyring_1.0-1_all.deb \
     && rm cuda-keyring_1.0-1_all.deb \
     && apt-get update -y \
     && apt-get install -y --no-install-recommends libcudnn9-cuda-12=9.1.0.* \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# 5) Install Nsight Compute
RUN wget -O nsight-compute.run https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2025_2_0/nsight-compute-linux-2025.2.0.11-35613519.run \
     && chmod +x nsight-compute.run \
     && ./nsight-compute.run -- -noprompt \
     && rm nsight-compute.run

# 6) Install Nsight Systems
RUN wget -O nsight-sys.run https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/NsightSystems-linux-public-2025.3.1.90-3582212.run \
     && chmod +x nsight-sys.run \
     && ./nsight-sys.run -- -noprompt \
     && rm nsight-sys.run