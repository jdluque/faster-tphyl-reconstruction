#!/bin/bash

for algorithm in 1 9 10 11
do
  sbatch --job-name b${algorithm}_melanoma -- batch_melanoma_exps.sh $algorithm
done
