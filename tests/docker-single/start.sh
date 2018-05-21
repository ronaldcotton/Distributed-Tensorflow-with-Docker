#!/bin/bash
# get host and user id outside of docker
export HUID=`id -u`
export HGID=`id -g`

docker run -t -d -v ${PWD}/tmp:/tmp --name tfworker0 ubuntu/tensorflow

# set group and user id of shared folder
docker exec -i tfworker0 chown ${HGID}:${HUID} /tmp
docker exec -i tfworker0 python /tmp/mnist_softmax_modified.py
