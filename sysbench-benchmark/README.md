
## bench.py ##
-----------------
*Use sysbench 0.4.12 for this software to work properly, newer versions use
different parameters*
Outputs machine info and runs sysbench CPU and FileIO benchmark for
determining baseline performance on system.  Benchmark data can be doun

### upgrade pip ###

pip3 install --upgrade pip

### installing sysbench ###

#### debian/ubuntu: ####
sudo apt-get install sysbench

#### MacOSX ####
brew install sysbench

#### Windows10 #### _install Windows Feature - Linux Subsystem_
sudo apt-get install sysbench

### other articles about sysbench: ###
http://imysql.com/wp-content/uploads/2014/10/sysbench-manual.pdf
https://anothermysqldba.blogspot.com/2013/05/benchmarking-mysql-cpu-file-io-memory.html
http://blog.siphos.be/2013/04/comparing-performance-with-sysbench-part-2/
