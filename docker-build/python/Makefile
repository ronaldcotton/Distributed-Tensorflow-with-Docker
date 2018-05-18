#!/bin/bash
export TF_BRANCH=r1.8
export PYTHON=3

export HUID=`id -u`
export HGID=`id -g`

.DEFAULT_GOAL := help
NAME := tensorflow-python
TAG := latest
IMG := ${NAME}:${TAG}
LATEST := ${NAME}:latest

.DEFAULT help:
	@echo "Docker Makefile"
	@echo "---------------"
	@echo "\tmake build"
	@echo "\t\tBuilds Tensorflow wheel for installing with pip"
	@echo "\t\tor outside the container."
	@echo "\tmake run"
	@echo "\t\tLogs into shell of Docker."
	@echo "\tmake attach"
	@echo "\t\treattached detached image (^p^q to detach in container)"
	@echo "\tmake pause"
	@echo "\t\tWhen a docker container is not connected, pause container"
	@echo "\tmake unpause"
	@echo "\t\tTo use or delete a container, you must unpause container"
	@echo "\tmake clean"
	@echo "\t\tRemoves docker image and deletes Tensorflow Wheel, if exists."
	@echo "\tmake cleanall"
	@echo "\t\tCleans all containers exited or not running"
build:
	docker build --build-arg PYTHON=${PYTHON} --build-arg TF_BRANCH=${TF_BRANCH} --build-arg huid=${HUID} --build-arg hgid=${HGID} --build-arg hgidname=${USER} -t ${IMG} .
	docker tag ${IMG} ${LATEST}
run:
	-mkdir -p host
	docker run -i -t -v${PWD}/host:/host -p 6006:6006 -p 8888:8888 --name ${NAME} ${IMG} /bin/bash
clean:
	-docker rmi -f ${IMG}
	-rm -rf host/*
cleanall:
	docker system prune -a
attach:
	docker attach ${NAME}
pause:
	docker container pause ${NAME}
unpause:
	docker container unpause ${NAME}
