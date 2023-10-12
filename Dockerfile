FROM nvidia/cuda:12.1.0-base-ubuntu20.04

RUN apt update -y
RUN apt upgrade -y
RUN apt install python3-pip -y
RUN apt install python-is-python3

WORKDIR /workspace
ADD . /workspace

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install .
ENTRYPOINT ["python" , "basic_nerf/main.py"]