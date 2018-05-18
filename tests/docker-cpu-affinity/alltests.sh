#!/bin/bash
# runs all tests
# from workers 1 to 24

for workers in `seq 1 24`; do
    for ps in `seq 1 3`; do
        python generateBashScripts.py --workers $workers --ps $ps
        ./start.sh
        ./stop.sh
        rm start.sh stop.sh
    done
done
