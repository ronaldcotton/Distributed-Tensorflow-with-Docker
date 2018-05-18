# tensorflow-dist-MNIST
# ronald.cotton@wsu.edu

FROM ubuntu:16.04
# vs. latest incase install method changes
WORKDIR /opt

# update machine
RUN apt-get update && apt-get upgrade -y \
	&& apt-get install -y python-pip python-dev python-setuptools unzip wget time iputils-ping gcc
RUN apt-get clean && apt-get autoremove

RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# currently the only version that works with tensorflow 1.8.0
RUN pip install h5py==2.8.0rc1

RUN pip install protobuf scipy tensorflow==1.8.0 keras psutil


#COPY <srchost> <dstcontainer>
COPY ./dist.py /opt/
#ADD contents to dst
RUN mkdir /opt/MNIST-data
# ADD MNIST-data /opt/MNIST-data

WORKDIR "/opt"
