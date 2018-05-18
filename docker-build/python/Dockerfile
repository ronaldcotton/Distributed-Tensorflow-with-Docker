# top to lines used to set permission for Tensorflow wheel, if needed
# otherwise defaults to root:root on system
# example to override default: docker build --build-args TF_BRANCH = r1.6

FROM debian:latest
MAINTAINER Ron Cotton <ronald.cotton@wsu.edu>

ARG huid
ARG hgid
ARG hgidname
ARG TF_BRANCH
ARG PYTHON

RUN echo "Host UID = '${huid}' Host GID = '${hgid}'\
 Tensorflow Branch = '${TF_BRANCH}'\
 Python Version = '${PYTHON}'"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# COPY files required inside docker.
COPY ./.makefile /root/

# System packages
RUN apt-get update --fix-missing \
 && apt-get upgrade -y \
 && apt-get install -y wget bzip2 curl default-jdk gnupg git-core nano tmux procps build-essential \
 && apt-get clean \
 && apt-get autoremove \
 && rm -rf /var/lib/apt/lists/*

# get tensorflow - benchmarks is optional
WORKDIR /root
RUN git clone -b $TF_BRANCH https://github.com/tensorflow/tensorflow
RUN git clone https://github.com/tensorflow/benchmarks.git

RUN export PATH="${PATH}:${HOME}/bin"

# makes .bashrc for logging in
RUN echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc
RUN echo 'alias tfbuild="make -f .makefile"' >> ~/.bashrc
RUN echo 'chown -R ${huid}:${hgid} /host' >> ~/.bashrc
RUN echo 'echo "To Build tensorflow wheel and install - tfbuild doall"' >> ~/.bashrc
RUN echo 'echo "For help with commands - tfbuild help"' >> ~/.bashrc

# Install miniconda to /miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda${PYTHON}-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -p /miniconda -b
RUN rm ~/miniconda.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda \
 && conda clean --all

# set up pip
RUN pip --no-cache-dir install --upgrade \
        pip setuptools \
 && pip --no-cache-dir install wheel \
 && pip --no-cache-dir install numpy \
 && pip --no-cache-dir install psutil \
 && pip --no-cache-dir install keras \
 && pip --no-cache-dir install h5py==2.8.0rc1

# Add Bazel distribution URI as a package source (one time setup)
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -

# install bazel
RUN apt-get update --fix-missing \
 && apt-get install -y bazel \
 && apt-get upgrade bazel \
 && apt-get clean \
 && apt-get autoremove \
 && rm -rf /var/lib/apt/lists/*

# creates same user as the one running Dockerfile
#RUN mkdir /host
#RUN groupadd -g ${hgid} ${hgidname} && useradd -r -u ${huid} -g ${hgid} ${hgidname} && chown -R ${hgidname} /host
RUN mkdir /host
RUN mkdir /host/logs
RUN mkdir /host/data
RUN mkdir /host/save
#RUN groupadd -g ${hgid} ${hgidname} && useradd -r -u ${huid} -g ${hgid} ${hgidname} && chown -R ${huid}:${hgid} /host

# Port for TensorBoard
EXPOSE 6006
# Prt for IPython Notebook
EXPOSE 8888
CMD [ "/bin/bash" ]
