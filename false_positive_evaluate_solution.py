import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", required=True)
parser.add_argument("--original", required=True)
parser.add_argument("--solution", required=True)
parser.add_argument("--algorithm", type=int, required=True)
parser.add_argument("--runtime", type=float, required=True)
parser.add_argument("--results", required=True)
args = parser.parse_args()

# Determine fp_probability
filename = os.path.basename(args.data_file)
if ".false_positives" in filename:
    fp_probability = float(filename.split("-fp_")[1].split("-")[0])
else:
    fp_probability = 0.0

# Determine the sim_name and before noise file
if ".false_positives" in filename:
    sim_name = filename.replace(".false_positives", "").replace(f"-fp_{fp_probability}", "-fp_0")
else:
    sim_name = filename.replace(".after_noise", "")
before_noise_file = sim_name + ".before_noise"

# Read in the before noise data
before_noise_df = pd.read_csv(os.path.join(args.original, before_noise_file), sep="\t", header=None)
before_noise_df = before_noise_df.iloc[:, 1:].iloc[1:, :].astype(int)

# Read in the solution data
solution_file = os.path.join(args.solution, os.path.splitext(filename)[0] + ".CFMatrix")
solution_df = pd.read_csv(solution_file, sep="\t", header=None)
solution_df = solution_df.iloc[:, 1:].iloc[1:, :].astype(int)

# Comparison metric 1: Number of flips (L1 distance)
l1_distance = np.sum(np.abs(solution_df.values - before_noise_df.values))

# Comparison metric 2: what percentage of pairs in the ground truth that have ancestor-descendant relationship are also ancestor-descendant in our solution matrix
ancestor_descendant_pair_count = 0
consistent_pair_count = 0
for i in range(before_noise_df.shape[1]): # hypothetical decendant
    for j in range(i + 1, before_noise_df.shape[1]): # hypothetical ancestor
        if not np.any(before_noise_df.iloc[:, i].values.view(bool) & ~before_noise_df.iloc[:, j].values.view(bool)): # Cell that has i but not j
            ancestor_descendant_pair_count += 1
            if not np.any(solution_df.iloc[:, i].values.view(bool) & ~solution_df.iloc[:, j].values.view(bool)):
                consistent_pair_count += 1

# False positive and false negative counts (from the data file)

# read data file
generated_df = pd.read_csv(args.data_file, sep="\t", header=None)
generated_df = generated_df.iloc[:, 1:].iloc[1:, :].astype(int)
num_false_positives = (generated_df > before_noise_df).sum().sum()
num_existing_false_negatives = (generated_df < before_noise_df).sum().sum()

# Append to CSV
results_df = pd.DataFrame([{
    "sim_name": sim_name,
    "algorithm": args.algorithm,
    "fp_probability": fp_probability,
    "num_rows": before_noise_df.shape[0],
    "num_cols": before_noise_df.shape[1],
    "num_existing_false_negatives": num_existing_false_negatives,
    "num_false_positives": num_false_positives,
    "l1_distance": l1_distance,
    "total_ancestor_descendant_pairs": ancestor_descendant_pair_count,
    "consistent_pair_count": consistent_pair_count,
    "filename": filename,
    "runtime": args.runtime
}])
results_df.to_csv(args.results, mode="a", header=False, index=False)
