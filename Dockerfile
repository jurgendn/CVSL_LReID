FROM python:3.10.13-slim-bookworm 
RUN apt-get update
RUN apt-get install -y cmake build-essential

# These command will be used to install
# - torch==1.13, torchvision
# - opencv-python
# - pytorch-lightning
# - torchmetrics

COPY requirements.txt /tmp/requirements.txt


RUN pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /tmp/requirements.txt
