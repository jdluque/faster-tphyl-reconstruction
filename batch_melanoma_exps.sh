#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --qos=medium
#SBATCH --mem=64gb
#SBATCH --output=phiscs.out.%j
#SBATCH --error=phiscs.out.%j
#SBATCH --cpus-per-task 8

algorithm=$1
file="real/melanoma20.tsv"

srun lscpu | grep 'Model name'

srun --time=2-00:00:00 -- python main.py --b $algorithm --input $file --output solutions/b$algorithm --log b${algorithm}.log
