#!/bin/bash
# get host and user id outside of docker
export HUID=`id -u`
export HGID=`id -g`

docker run -t -d -v ${PWD}/opt:/opt --name tfworker0 ubuntu/tensorflow

# set group and user id of shared folder
docker exec -i tfworker0 chown ${HGID}:${HUID} /opt
docker exec -i tfworker0 python /opt/mnist_softmax_modified.py
