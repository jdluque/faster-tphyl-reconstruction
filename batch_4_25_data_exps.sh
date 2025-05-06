#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --qos=medium
#SBATCH --mem=64gb
#SBATCH --output=phiscs.out.%j
#SBATCH --error=phiscs.out.%j
#SBATCH --cpus-per-task 8

algorithm=$1

srun lscpu | grep 'Model name'

times_file="experiments/times.csv"
log_file="skipped.log"

alg_names=(
  "TwoSat full"
  "TwoSat compact"
  "LP GLOP full rewrite"
  "LP PDLP partial rewrite"
  "LP Gurobi partial rewrite"
  "LP PDLP full rewrite"
  "Vertex Cover"
  "Maximum Weight Matching"
  "Hybrid LP Gurobi partial rewrite"
  "Hybrid LP PDLP full rewrite"
  "Hybrid LP PDLP partial rewrite"
)

alg_index=$((algorithm - 1))
alg_name="${alg_names[$alg_index]}"

# Ensure skipped.log exists
touch "$log_file"

for file in $(ls -Sr data_4_28_25/*.SC.after_noise);
do
  dataset_name=$(basename "$file") # removes the .after_noise part

  # Check if this algorithm and dataset combination has done
  if grep -q "^$alg_name,$dataset_name," "$times_file";
  then
    echo "$(date): $alg_name skipped $dataset_name" >> "$log_file"
    continue
  fi

  srun --time=2:00:00 -- python main.py --b $algorithm --input $file --output solutions/b$algorithm --log b${algorithm}.log
done