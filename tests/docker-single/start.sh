#!/bin/bash
docker run -t -d -v ./tmp:/tmp --name tfworker0 ubuntu/tensorflow
docker exec -i tfworker0 python mnist_softmax_modified.py
