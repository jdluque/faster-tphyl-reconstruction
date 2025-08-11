# Imports
import os
import pandas as pd
import numpy as np
import subprocess

# Constants
base_folder = "false_positive/"
original_data_folder = base_folder + "original_data/" # "Juan_Arjun-Apr_28_2025/"
generated_data_folder = base_folder + "generated_data/"
solution_folder = base_folder + "solutions/"
results_csv_file = base_folder + "false_positive_results.csv"
fp_probability = 0.002 # .2% of the total number of elements
np.random.seed(42)  # For reproducibility (this can be changed)
algorithms_to_run = [11]  # List of algorithms to run
run_no_false_positive_eval = True  # Set to False to skip this evaluation

# Make sure the folders exist
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
if not os.path.exists(original_data_folder):
    raise FileNotFoundError(f"Original data folder {original_data_folder} does not exist.")
if not os.path.exists(generated_data_folder):
    os.makedirs(generated_data_folder)
if not os.path.exists(solution_folder):
    os.makedirs(solution_folder)

# Create the results CSV file if it doesn't exist
if not os.path.exists(results_csv_file): # NOTE: IT KEEPS OLD RECORDS
    results_df = pd.DataFrame(columns=[
        "sim_name", "algorithm", "fp_probability", "num_rows", "num_cols", "num_existing_false_negatives", "num_false_positives",
        "l1_distance", "total_ancestor_descendant_pairs", "consistent_pair_count", "runtime"
    ])
    results_df.to_csv(results_csv_file, index=False)

# Create a list of all the simulations to run
simulation_files = [
    f.rsplit('.after_noise', 1)[0]
    for f in os.listdir(original_data_folder)
    if f.endswith('.after_noise') and os.path.isfile(os.path.join(original_data_folder, f))
]

# Start processing each simulation
print(f"Starting processing of simulations... ({len(simulation_files)} simulations found)")
for i, sim_file in enumerate(simulation_files):
    print(f"{pd.Timestamp.now().strftime('%H:%M:%S')}: Processing simulation {i + 1}/{len(simulation_files)}: {sim_file}")

    # Read in original data
    before_noise_df = pd.read_csv(original_data_folder + sim_file + ".before_noise", sep='\t', header=None)
    after_noise_df = pd.read_csv(original_data_folder + sim_file + ".after_noise", sep='\t', header=None)
    new_data = after_noise_df.copy() # for later
    before_noise_df = before_noise_df.iloc[:, 1:].iloc[1:, :].astype(int)
    after_noise_df = after_noise_df.iloc[:, 1:].iloc[1:, :].astype(int)

    # Add false positives
    num_false_positives = 0 # count of false positives added
    num_existing_false_negatives = 0 # count of existing false negatives in the original data
    for i in range(before_noise_df.shape[0]):
        for j in range(before_noise_df.shape[1]):
            if before_noise_df.iloc[i, j] == 1 and after_noise_df.iloc[i, j] == 0: # false negative
                num_existing_false_negatives += 1
            if before_noise_df.iloc[i, j] == 0 and after_noise_df.iloc[i, j] == 1: # false positive
                raise ValueError(f"{sim_file} already has false positives, which is unexpected.")
            if before_noise_df.iloc[i, j] == 0 and after_noise_df.iloc[i, j] == 0:
                # Randomly decide to add a false positive
                if np.random.rand() < fp_probability:
                    new_data.iloc[i + 1, j + 1] = str(after_noise_df.iloc[i, j] + 1)
                    # Note: +1 since there are row and column titles in the original data

    # Save the new data with false positives
    new_data_file = generated_data_folder + sim_file + ".false_positives" # NOTE: How the file is named
    new_data_file = new_data_file.replace("-fp_0-", f"-fp_{fp_probability}-")
    new_data.to_csv(new_data_file, sep='\t', header=False, index=False)

    # Run the algorithm on the new data
    times = []
    after_noise_file = original_data_folder + sim_file + ".after_noise" # no false positives
    for alg in algorithms_to_run:
        for data_source in range(1 + int(run_no_false_positive_eval)): # 0: false positive data, 1: original "after_noise" data
            time_temp = pd.Timestamp.now()
            subprocess.run(["conda", "run", "-n", "PHISCS", "python3", "main.py", "-i", 
                            new_data_file if data_source == 0 else after_noise_file,
                            "-o", solution_folder, "-b", str(alg)], check=True, stdout=subprocess.DEVNULL)
            time = (pd.Timestamp.now() - time_temp).total_seconds()
            times.append(time)

    # Evaluate the solution
    for alg in algorithms_to_run:
        for data_source in range(1 + int(run_no_false_positive_eval)): # 0: false positive data, 1: original "after_noise" data

            # Read the output file
            base_file = (new_data_file if data_source == 0 else after_noise_file).split('/')[-1].rsplit('.', 1)[0]
            noisy_solution_file = solution_folder + base_file + ".CFMatrix"
            noisy_solution_df = pd.read_csv(noisy_solution_file, sep='\t', header=None)
            noisy_solution_df = noisy_solution_df.iloc[:, 1:].iloc[1:, :].astype(int)

            # Comparison metric 1: Number of flips (L1 distance)
            l1_distance = np.sum(np.abs(noisy_solution_df.values - before_noise_df.values))

            # Comparison metric 2: what percentage of pairs in the ground truth that have ancestor-descendant relationship are also ancestor-descendant in our solution matrix
            ancestor_descendant_pair_count = 0
            consistent_pair_count = 0
            for i in range(before_noise_df.shape[1]): # hypothetical decendant
                for j in range(i + 1, before_noise_df.shape[1]): # hypothetical ancestor
                    if not np.any(before_noise_df.iloc[:, i].values.view(bool) & ~before_noise_df.iloc[:, j].values.view(bool)): # Cell that has i but not j
                        ancestor_descendant_pair_count += 1
                        if not np.any(noisy_solution_df.iloc[:, i].values.view(bool) & ~noisy_solution_df.iloc[:, j].values.view(bool)):
                            consistent_pair_count += 1

            # Store the results (append to CSV don't read the whole file)
            results_df = pd.DataFrame([{
                "sim_name": sim_file,
                "algorithm": alg,
                "fp_probability": fp_probability if data_source == 0 else 0,
                "num_rows": before_noise_df.shape[0],
                "num_cols": before_noise_df.shape[1],
                "num_existing_false_negatives": num_existing_false_negatives,
                "num_false_positives": num_false_positives if data_source == 0 else 0,
                "l1_distance": l1_distance,
                "total_ancestor_descendant_pairs": ancestor_descendant_pair_count,
                "consistent_pair_count": consistent_pair_count,
                "runtime": times[algorithms_to_run.index(alg) * (1 + int(run_no_false_positive_eval)) + data_source]
            }])
            results_df.to_csv(results_csv_file, mode='a', header=False, index=False)

# End
print("Done with false positive evaluation!")
