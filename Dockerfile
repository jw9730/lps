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
RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
