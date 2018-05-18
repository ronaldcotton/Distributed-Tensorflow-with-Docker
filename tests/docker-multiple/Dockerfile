# tensorflow-dist-MNIST
# ronald.cotton@wsu.edu

FROM ubuntu:16.04
# vs. latest incase install method changes
WORKDIR /opt

# update machine
RUN apt-get update && apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends python-pip python-dev python-setuptools unzip wget time iputils-ping
RUN apt-get clean && apt-get autoremove

RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install protobuf scipy h5py tensorflow keras

# May be outdated soon
# RUN pip install --ignore-installed --upgrade https://github.com/roncotton/tensorflow-wheels-r1.6/blob/master/python3/tensorflow-1.6.0-cp36-cp36m-linux_x86_64.whl?raw=true

WORKDIR "/opt"
