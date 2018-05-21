Tests
——---
This set consists of three tests: docker-cpu-affinity, docker-multiple, and
docker-single.  All three of these models consist of running a simple neural
network with one hidden layer of 100 elements, sharing the exact same model all
contained within 1 (or more) docker containers.

Docker-single, creates a single docker container and runs a non-distributed test
of Tensorflow.  This is the base-case and was created to differentiate distributed
runtime and execution.

Docker-multiple generates multiple docker container and runs a Tensorflow RPC
(ClusterSpec) over a docker network, assigning parameter servers with the ip of
10.20.30.1X and the workers with an ip of 10.20.30.1XX using port 2222 to
transfer communication between the parameter servers and the workers.  While
this simulates a real network accurately, the overhead of running several
servers on the same machine is large.  With minor adjustments, this can be
implemented over a real network.

Docker-cpu-affinity generates a single docker container and runs a Tensorflow
RPC (Clusterspec) over a docker network, all using the same ip, but given a
port.  The workload is divided up into specific CPUS, using CPU Affinity.
Any remaining workload is given to the chief process, worker0.

The resulting tests can be found in /collected-data.

The MNIST dataset is placed into each test set so that the test set is not
needed to be downloaded.

Requirements:
  Docker
  Python Environment
  Bash

If docker requires 'sudo' access to operate, get into a sudo environment first: sudo -s.

For docker-cpu-affinity and docker-multiple - run ./alltest.sh.
For docker-single, type ./start.sh to begin testing and ./stop.sh to end testing.

.
├── docker-cpu-affinity
│   ├── Dockerfile
│   ├── alltests.sh
│   ├── generateBashScripts.py
│   └── opt
│       ├── MNIST-data
│       │   ├── t10k-images-idx3-ubyte.gz
│       │   ├── t10k-labels-idx1-ubyte.gz
│       │   ├── train-images-idx3-ubyte.gz
│       │   └── train-labels-idx1-ubyte.gz
│       └── dist.py
├── docker-multiple
│   ├── Dockerfile
│   ├── alltests.sh
│   ├── generateBashScripts.py
│   └── opt
│       ├── MNIST-data
│       │   ├── t10k-images-idx3-ubyte.gz
│       │   ├── t10k-labels-idx1-ubyte.gz
│       │   ├── train-images-idx3-ubyte.gz
│       │   └── train-labels-idx1-ubyte.gz
│       └── dist.py
└── docker-single
    ├── Dockerfile
    ├── opt
    │   ├── MNIST-data
    │   │   ├── t10k-images-idx3-ubyte.gz
    │   │   ├── t10k-labels-idx1-ubyte.gz
    │   │   ├── train-images-idx3-ubyte.gz
    │   │   └── train-labels-idx1-ubyte.gz
    │   └── mnist_softmax_modified.py
    ├── start.sh
    └── stop.sh
