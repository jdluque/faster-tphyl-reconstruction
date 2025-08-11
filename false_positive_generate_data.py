import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--original", required=True)
parser.add_argument("--generated", required=True)
parser.add_argument("--fp_probs", nargs="+", type=float, required=True)
args = parser.parse_args()

# For reproducibility (this can be changed)
np.random.seed(42)

# Create a list of all the simulations to run
simulation_files = [
    f.rsplit(".after_noise", 1)[0]
    for f in os.listdir(args.original)
    if f.endswith(".after_noise") and os.path.isfile(os.path.join(args.original, f))
]

# Create new data files
for fp_prob in args.fp_probs:
    for sim_file in simulation_files:
        generated_file = os.path.join(args.generated, sim_file.replace("-fp_0-", f"-fp_{fp_prob}-") + ".false_positives") # Replace the fp probability in the filename
        if os.path.exists(generated_file):
            continue  # Skip if already generated

        # Read in original data
        before_df = pd.read_csv(os.path.join(args.original, sim_file + ".before_noise"), sep="\t", header=None)
        after_df = pd.read_csv(os.path.join(args.original, sim_file + ".after_noise"), sep="\t", header=None)
        new_data = after_df.copy() # for later
        before_df = before_df.iloc[:, 1:].iloc[1:, :].astype(int)
        after_df = after_df.iloc[:, 1:].iloc[1:, :].astype(int)

        # Add false positives
        for i in range(before_df.shape[0]):
            for j in range(before_df.shape[1]):
                if before_df.iloc[i, j] == 0 and after_df.iloc[i, j] == 1: # false positive
                    raise ValueError(f"{sim_file} already has false positives, which is unexpected.")
                if before_df.iloc[i, j] == 0 and after_df.iloc[i, j] == 0:
                    # Randomly decide to add a false positive
                    if np.random.rand() < fp_prob:
                        new_data.iloc[i + 1, j + 1] = str(after_df.iloc[i, j] + 1)
                        # Note: +1 since there are row and column titles in the original data

        # Save the new data with false positives
        new_data.to_csv(generated_file, sep="\t", header=False, index=False)