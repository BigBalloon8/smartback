FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel

RUN pip install termcolor
RUN pip install einops
RUN pip install packaging
#RUN pip install causal-conv1d>=1.2.0
RUN pip install mamba-ssm
RUN pip install tqdm
RUN pip install click
RUN pip install fft-conv-pytorch
RUN pip install torch-summary
RUN apt update
RUN apt-get install libglib2.0-dev -y
RUN apt install wget -y 
RUN wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_2/nsightsystems-linux-cli-public-2024.2.1.106-3403790.deb
RUN dpkg -i nsightsystems-linux-cli-public-2024.2.1.106-3403790.deb
#RUN git clone https://github.com/state-spaces/mamba.git; cd mamba; pip pip install .

COPY . .