FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

LABEL maintainer="Evgeny Nizhibitsky <nizhib@github.com>, Guodong Xu <memoiry@outlook.com>"

# Install ubuntu packages

COPY sources.list /etc/apt/sources.list
RUN apt-get clean &&\
    apt-get update --allow-unauthenticated && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        python3-pip \
        python3-setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY . .

# Run the service

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD python3 -u app.py
