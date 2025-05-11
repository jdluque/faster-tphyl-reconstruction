#!/bin/bash

for algorithm in 2 9 10 11
do
  sbatch --job-name b${algorithm}_melanoma -- batch_melanoma_exps.sh $algorithm
done
