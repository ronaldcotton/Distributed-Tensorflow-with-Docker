/collected-data
  includes all run test data, including spreadsheets and
  documentation.

/docker-build
  contains a Docker build environment implementing Ubuntu with a
  Python Conda environment, and all the programs needed to build a Tensorflow binary.
  This container is to simplify making an optimized Tensorflow wheel.

/docs
  contains the final report and all documentation collected.

/sysbench-benchmark
  is the benchmarking suite which runs tests on CPU, Threads,
  and FileIO.  To run, simply run bench.py.  Note: 16GB of free harddrive space is
  required to run the benchmark.

/tests
  Running the same model, this contains both the distributed and non-distributed
  tests.

/extra
  Incomplete applications
  With initial research with tflearn, build a simple hyper-parameter optimizer.

  When Aaron Goin and I were going to combine our distributed application, I
  noticed that many of the structures of Tensorflow.js were similiar to Keras.
  Built a simple Tensorflow/Keras MNIST DNN/CNN, which had a separate model.py
  which would allow reprogramming.  Even has Tensorboard capabilities.
  Time didn't allow this project to finish.
