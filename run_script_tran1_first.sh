#!/bin/bash

epsilons=( 0.1 1 10 )

for eps in "${epsilons[@]}"
do
    for index in {1234..1243}
    do
        nohup python3 run_main_lin_$1.py tran1 $eps $index &
        sleep .5
    done
done