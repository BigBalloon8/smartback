FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN pip install termcolor
RUN pip install einops
RUN pip install causal-conv1d>=1.2.0
RUN pip install mamba-ssm

COPY . .