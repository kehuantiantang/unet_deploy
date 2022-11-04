#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#FROM nvidia/cuda:10.2-devel-ubuntu18.04
FROM nvidia/cuda:11.1.1-devel-ubuntu18.04
LABEL maintainer="youjie80@gmail.com>"

# Deal with pesky Python 3 encoding issue
ENV DEBIAN_FRONTEND noninteractive

# User should be refactored instead of hard coded jovyan
ARG NB_USER=jovyan
ARG UID

USER root

ENV DEBIAN_FRONTEND noninteractive
ENV NB_USER $NB_USER

# uid should obtain the value by "id -u"
ENV NB_UID 1000
#ENV NB_UID $UID
ENV HOME /home/$NB_USER
ENV NB_PREFIX /
ENV PATH $HOME/.local/bin:$PATH

# Use bash instead of sh
SHELL ["/bin/bash", "-c"]

#RUN apt-key del 7fa2af80
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update &&\
    apt-get install -y software-properties-common \
    sudo \
    build-essential \
    git \
    wget \
    vim \
    curl \
    zip \
    zlib1g-dev \
    unzip \
    locales \
    pkg-config \
    swig \
    apt-transport-https \
    lsb-release \
    ffmpeg \
    libsm6 \
    libxext6 \
    ca-certificates &&\
    curl -sL https://deb.nodesource.com/setup_12.x | bash - &&\
    apt install -y nodejs &&\
#    ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
#    ln -s /usr/bin/python3.8 /usr/local/bin/python3 &&\

    apt-get clean

# setup conda environment in docker
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O /tmp/anaconda.sh && \
        /bin/bash /tmp/anaconda.sh -b -p /opt/conda && \
        rm /tmp/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy
ENV PATH /opt/conda/bin:$PATH

COPY ./environment.yaml /tmp/environment.yaml
RUN conda update conda \
    && conda env create --name mmsegmentation -f /tmp/environment.yaml

RUN echo "conda activate mmsegmentation" >> ~/.bashrc
ENV PATH /opt/conda/envs/mmsegmentation/bin:$PATH
ENV CONDA_DEFAULT_ENV $mmsegmentation

#RUN chmod 777 -R /opt/conda

RUN ln -s /opt/conda/envs/mmsegmentation/bin/python /usr/local/bin/python
#    ln -s /usr/bin/python3.8 /usr/local/bin/python3 &&\
#RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


#COPY requirements.txt /tmp
#RUN pip --no-cache-dir install -r  /tmp/requirements.txt

#RUN apt-get update &&\
#    apt-get install -y software-properties-common \
#    sudo \
#    build-essential \
#    git \
#    wget \
#    vim \
#    curl \
#    zip \
#    zlib1g-dev \
#    unzip \
#    locales \
#    pkg-config \
#    libgl-dev \
#    libblas-dev \
#    liblapack-dev \
#    python3-tk \
#    python3-wheel \
#    graphviz \
#    libhdf5-dev \
#    python3.8 \
#    python3.8-dev \
#    python3.8-distutils \
#    swig \
#    apt-transport-https \
#    lsb-release \
#    ca-certificates &&\
#    # obtain latest of nodejs
#    curl -sL https://deb.nodesource.com/setup_12.x | bash - &&\
#    apt install -y nodejs &&\
#    apt-get clean &&\
#    ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
#    ln -s /usr/bin/python3.8 /usr/local/bin/python3 &&\
#    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
#    python3 get-pip.py &&\
#    rm get-pip.py &&\
#    # best practice to keep the Docker image lean
#    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


#RUN apt-get update && apt-get -y install libgl1-mesa-glx


# Install essential Python packages
#RUN python3 -m pip --no-cache-dir install \
#    blackcellmagic\
#    pytest \
#    pytest-cov \
#    numpy \
#    matplotlib \
#    scipy \
#    pandas \
#    scikit-learn \
#    scikit-image \
#    seaborn \
#    graphviz \
#    gpustat \
#    h5py \
#    gitpython \
#    ptvsd \
#    Pillow==6.1.0 \
#    opencv-python

# Install Nodejs for jupyterlab-manager
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
RUN apt-get update && apt-get install -yq --no-install-recommends \
  nodejs \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

ENV DOCKER_CREDENTIAL_GCR_VERSION=1.4.3
RUN curl -LO https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${DOCKER_CREDENTIAL_GCR_VERSION}/docker-credential-gcr_linux_amd64-${DOCKER_CREDENTIAL_GCR_VERSION}.tar.gz && \
    tar -zxvf docker-credential-gcr_linux_amd64-${DOCKER_CREDENTIAL_GCR_VERSION}.tar.gz && \
    mv docker-credential-gcr /usr/local/bin/docker-credential-gcr && \
    rm docker-credential-gcr_linux_amd64-${DOCKER_CREDENTIAL_GCR_VERSION}.tar.gz && \
    chmod +x /usr/local/bin/docker-credential-gcr

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8


# RUN printf '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d

#sshd
#RUN echo "${NB_USER}:${NB_USER}" | chpasswd
#RUN echo 'root:password' | chpasswd
#RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
#RUN echo 'root:ubuntu' | chpasswd
# RUN echo "jovyan:jovyan" | chpasswd
EXPOSE 22

#RUN ssh-keygen -P "" -t dsa -f /etc/ssh/ssh_host_dsa_key
#RUN service ssh start
#ENTRYPOINT service ssh restart && bash
#CMD ["/usr/sbin/sshd","-D", "-o", "ListenAddress=0.0.0.0"]
#---------------------------------------------

# Create NB_USER user with UID=1000 and in the 'users' group
# but allow for non-initial launches of the notebook to have
# $HOME provided by the contents of a PV
RUN useradd -M -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown -R ${NB_USER}:users /usr/local/bin && \
    mkdir -p $HOME && \
    chown -R ${NB_USER}:users ${HOME} && \
    usermod -aG sudo ${NB_USER}
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


# Install Tini - used as entrypoint for container
RUN cd /tmp && \
    wget --quiet https://github.com/krallin/tini/releases/download/v0.18.0/tini && \
    echo "12d20136605531b09a2c2dac02ccee85e1b874eb322ef6baf7561cd93f93c855 *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini


#--------------------------------------------------------------------------------

# RUN mkdir $HOME/scripts && \
# 		cd $HOME/scripts && \
# 		mkdir converter_scripts && \
# 		mkdir inference_demos

# COPY converter_scripts/ scripts/converter_scripts/
# COPY inference_demos/ scripts/inference_demos/

#---------------------------------------------------------------------------------

# ------------------- pip library ----------------
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install \
        jupyter-console==6.0.0 \
        jupyterlab \
        grpcio \
        grpcio-tools \
        datajoint==0.12.7 \
        easydict==1.9 \
        protobuf==3.20.2

#RUN pip --no-cache-dir install torch==1.9.0+cu102 \
#    torchvision==0.10.0+cu102 torchaudio==0.9.0 \
#    -f https://download.pytorch.org/whl/torch_stable.html
   
RUN pip --no-cache-dir install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip --no-cache-dir install mmsegmentation

RUN pip --no-cache-dir install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html


# COPY --chown=test:users requirements.txt /tmp
# RUN pip3 --no-cache-dir install -r /tmp/requirements.txt

#RUN cd /home/${NB_USER}/model/jbnu/ && \
#    pip install -e . --user

# RUN git clone https://github.com/ZetaLx/mmsegmentation.git /home/jovyan/model/jbnu/mmsegmentation
# WORKDIR /home/jovyan/model/jbnu/mmsegmentation
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -e .


RUN docker-credential-gcr configure-docker && chown ${NB_USER}:users $HOME/.docker/config.json


# Configure container startup
EXPOSE 50051
EXPOSE 8888
USER jovyan

WORKDIR $HOME

# COPY --chown=jovyan:users ./start.sh $HOME
#RUN chmod 777 ${HOME}/start.sh
#CMD bash -c "cd /home/jovyan && ./start.sh & "


#ENTRYPOINT ["tini", "--"]
#CMD ["sh","-c", "jupyter lab --notebook-dir=/home/${NB_USER} --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX} --NotebookApp.max_buffer_size=10000000000 " ]
