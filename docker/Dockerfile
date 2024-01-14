FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \
    && apt-get update \
    && apt-get install python3 -y \
    && apt-get install python3-pip -y
 
RUN echo "==> Install dos2unix..." \
 && sudo apt-get install dos2unix -y 

RUN echo "==> Install langchain requirements.." \
 && pip install langchain

RUN echo "==> Install streamlit.." \
 && pip install streamlit

# Install transformers and torch
RUN echo "==> Install transformers for Phi-2 model..." \
 && pip install transformers 

# Install PyTorch compatible with CUDA
RUN echo "==> Install torch for Phi-2 model..." \
 && pip install torch --extra-index-url https://download.pytorch.org/whl/cu110

COPY /capophied /capophied/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
 && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]