#!/bin/bash

for algorithm in 2 8 9 11 12 13
do
  sbatch --job-name b${algorithm}_phiscs -- batch_4_25_data_exps.sh $algorithm
done
