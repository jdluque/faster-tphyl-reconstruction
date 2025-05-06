#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --qos=medium
#SBATCH --mem=64gb
#SBATCH --output=phiscs.out.%j
#SBATCH --error=phiscs.out.%j
#SBATCH --cpus-per-task 8

srun lscpu | grep 'Model name'

algorithm=10

for file in $(ls -Sr data_4_28_25/*.SC.after_noise);
do
  python main.py --b $algorithm --input $file --output solutions/b$algorithm --log b${algorithm}.log
done
