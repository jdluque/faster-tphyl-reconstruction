#!/bin/bash

for algorithm in 12
do
  sbatch --job-name b${algorithm}_phiscs -- batch_4_25_data_exps.sh $algorithm
done
