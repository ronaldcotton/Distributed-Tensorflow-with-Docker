# ronald.cotton@wsu.edu

FROM ubuntu:16.04
WORKDIR /opt

# update machine
RUN apt-get update && apt-get upgrade -y \
	&& apt-get install -y python-pip python-dev python-setuptools unzip wget time iputils-ping gcc
RUN apt-get clean && apt-get autoremove

# clean apt-get after installs
RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# force using latest release candidate to remove warning caused by tensorflow
RUN pip install h5py==2.8.0rc1
RUN pip install protobuf scipy tensorflow==1.8.0 keras psutil

WORKDIR "/opt"
