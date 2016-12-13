FROM ubuntu:16.04
MAINTAINER VK.COM

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        cmake \
        wget \
        libatlas-base-dev \
        git \
        sudo \
        unzip && \
    rm -rf /var/lib/apt/lists/*

ENV WORKSPACE /workspace
WORKDIR ${WORKSPACE}
VOLUME /workspace/mount

# Install Torch Lua
RUN git clone https://github.com/torch/distro.git ~/torch --recursive && \
    cd ~/torch; bash install-deps && \
    ./install.sh

# Install Android NDK
RUN wget https://dl.google.com/android/repository/android-ndk-r10e-linux-x86_64.zip  && \
    unzip android-ndk-r10e-linux-x86_64.zip -d /usr/local/share

# Set environment
ENV NDK_PATH=/usr/local/share/android-ndk-r10e
ENV PATH="${NDK_PATH}:${PATH}"

# Add sources
ADD . ${WORKSPACE}
CMD ["/bin/bash", "./generate_android_docker.sh"]
