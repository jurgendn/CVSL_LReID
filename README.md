# Contrastive Viewpoint-aware Shape Learning for Long-term Person Re-Identification

This repository contains the code and resources for the publication titled "Contrastive Viewpoint-aware Shape Learning for Long-term Person Re-Identification". The publication proposes a method for person re-identification that leverages body shape for matching people across long durations and viewpoint variations.

## Overview

Traditional approaches for Person Re-identification (ReID) rely heavily on modeling the appearance of persons. This measure is unreliable over longer durations due to the possibility for changes in clothing or biometric information. Furthermore, viewpoint changes significantly degrade the matching ability of these methods. In this paper, we propose “Contrastive Viewpoint-aware Shape Learning for Long-term Person Re-Identification” (CVSL) to address these challenges. Our method robustly extracts local and global texture-invariant human body shape cues from 2D pose using the Relational Shape Embedding branch, which consists of a pose estimator and a shape encoder built on a Graph Attention Network. To enhance the discriminability of the shape and appearance of identities under viewpoint variations, we propose Contrastive Viewpoint-aware Losses (CVL). CVL leverages contrastive learning to simultaneously minimize the intra-class gap under different viewpoints and maximize the inter-class gap under the same viewpoint. Extensive experiments demonstrate that our proposed framework outperforms state-of-the-art methods on long-term person Re-ID benchmarks.

![[Pasted image 20240311140842.png]]

## Installation

This project can be installed using either pip in a virtual environment or Docker Compose.

### Using pip with a virtual environment

It's recommended to use a virtual environment to isolate project dependencies:

```bash
python -m venv venv
source venv/bin/activate  # activate for Unix-based systems
# venv\Scripts\activate.bat  # activate for Windows
```

Install dependencies:
Navigate to the project directory and install the required libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
```

### Using Docker Compose

This project provides a docker-compose.yml file to set up the environment with Docker.

#### Prerequisites

Ensure you have Docker and Docker Compose installed on your system. You can find installation instructions on the official websites:

- Docker: https://www.docker.com/
- Docker Compose: https://docs.docker.com/compose/install/

If you want to run the training procedure with GPU support inside a container, please install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Configure the production repository

```bash
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Update the packages list from the repository

```bash
$ sudo apt-get update
```

Install the NVIDIA Container Toolkit packages

```bash
$ sudo apt-get install -y nvidia-container-toolkit
```

#### Build and run the environment:

Navigate to the project directory and run the following command to build the Docker image and start the services defined in docker-compose.yml:

```bash
docker-compose up -d
```

The **Dockerfile** will contain

```Dockerfile
FROM python:3.10.13-slim-bookworm
RUN apt-get update
RUN apt-get install -y cmake build-essential

COPY requirements.txt /tmp/requirements.txt

RUN pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r /tmp/requirements.txt
```

and the **docker-compose.yaml** will contain:

```yaml
version: "3.8"

services:
  dev:
    build:
      dockerfile: Dockerfile
      context: .
    container_name: "CVSL_ReID"
    working_dir: /home/working
    tty: true
    stdin_open: true
    volumes:
      - .:/home/working

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Note that we may require adjustments depending on your specific needs, for example:

- Mounting dataset
- Exporse port
- Select the **deploy** section with more devices. For more detail, see [GPU Support](https://docs.docker.com/compose/gpu-support/).
