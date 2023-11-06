FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# install pip
RUN apt update && apt install python3-pip -y

# install git
RUN apt install git -y

# clone repository
RUN mkdir -p /lps
RUN git clone https://github.com/jw9730/lps.git /lps
WORKDIR /lps

# install dependencies
RUN pip3 install numpy==1.23.5
RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install xformers==0.0.20 lightning==2.0.6 transformers==4.31.0
RUN pip3 install --pre timm==0.9.5
RUN pip3 install easydict==1.10 fastapi==0.100.1 wandb==0.15.8 plum-dispatch==2.1.1 scikit-learn==1.3.0 matplotlib==3.7.2 seaborn==0.12.2
RUN pip3 install datasets==2.14.3
RUN pip3 install torch_geometric==2.4.0
RUN pip3 install torch_scatter==2.1.2+pt20cu118 torch_sparse==0.6.18+pt20cu118 -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
RUN pip3 install pytorch3d==0.3.0
RUN pip3 install tensorboard==2.13.0
RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
