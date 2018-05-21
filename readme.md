/collected-data
  includes all run test data, including spreadsheets and
  documentation.

/docker-build
  contains a Docker build environment implementing Ubuntu with a
  Python Conda enviroment, and all the programs needed to build a Tensorflow binary.
  This container is to simplify making an optimized Tnesorflow wheel.

/docs
  contains the final report and all documentation collected.

/sysbench-benchmark
  is the benchmarking suite which runs tests on CPU, Threads,
  and FileIO.  To run, simply run bench.py.  Note: 16GB of free harddrive space is
  required to run the benchmark.

/tests
  Running the same model, this contains both the distributed and non-distributed
  tests.
