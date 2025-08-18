#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --qos=medium
#SBATCH --mem=64gb
#SBATCH --output=phiscs.out.%j
#SBATCH --error=phiscs.out.%j
#SBATCH --cpus-per-task 8

# ===== Constants =====
BASE_FOLDER="false_positive"
ORIGINAL_DATA_FOLDER="data_4_28_25" # "Juan_Arjun-Apr_28_2025/"
GENERATED_DATA_FOLDER="$BASE_FOLDER/generated_data"
SOLUTION_FOLDER="$BASE_FOLDER/solutions"
RESULTS_CSV_FILE="$BASE_FOLDER/false_positive_results.csv"

FP_PROBABILITIES=(0.0 0.001 0.0001 0.00001) # Probabilities for a false positive
ALGORITHMS_TO_RUN=(11 2) # List of algorithms to run
RUN_NO_FALSE_POSITIVE_EVAL=false  # Set to false to skip additional evaluation

srun lscpu | grep 'Model name'

# ===== Ensure folders exist =====
mkdir -p "$BASE_FOLDER" "$GENERATED_DATA_FOLDER" "$SOLUTION_FOLDER"

if [ ! -d "$ORIGINAL_DATA_FOLDER" ]; then
    echo "ERROR: Original data folder $ORIGINAL_DATA_FOLDER does not exist."
    exit 1
fi

# ===== Create results CSV if not exists =====
if [ ! -f "$RESULTS_CSV_FILE" ]; then
    echo "sim_name,algorithm,fp_probability,num_rows,num_cols,num_existing_false_negatives,num_false_positives,l1_distance,total_ancestor_descendant_pairs,consistent_pair_count,filename,runtime" > "$RESULTS_CSV_FILE"
fi

# ===== Step 1: Generate false positive data =====
echo "Generating new data files..."
srun --time=1:00:00 -- python3 false_positive_generate_data.py \
    --original "$ORIGINAL_DATA_FOLDER" \
    --generated "$GENERATED_DATA_FOLDER" \
    --fp_probs "${FP_PROBABILITIES[@]}"

echo "Data generation complete."

# ===== Step 2: Run algorithms =====
FILES_TO_RUN=()

# Include generated false positive files
while IFS= read -r -d '' file; do
    FILES_TO_RUN+=("$file")
done < <(find "$GENERATED_DATA_FOLDER" -type f -name "*.false_positives" -print0)

# Optionally include after_noise files from original data
if [ "$RUN_NO_FALSE_POSITIVE_EVAL" = true ]; then
    while IFS= read -r -d '' file; do
        FILES_TO_RUN+=("$file")
    done < <(find "$ORIGINAL_DATA_FOLDER" -type f -name "*.after_noise" -print0)
fi

# Sort files using Python script
    # Ordering: m asc, fn_percentage acs, n desc, sim_number asc, fp_probability acs
    # Objective: The same simulation, all fp values should be run continually
    # Objective: Go from easiest to hardest datasets
SORTED_FILES_TO_RUN=()
while IFS= read -r file; do
    SORTED_FILES_TO_RUN+=("$file")
done < <(printf "%s\n" "${FILES_TO_RUN[@]}" | python3 false_positive_sort_files.py)
FILES_TO_RUN=("${SORTED_FILES_TO_RUN[@]}")

# Print status
echo "Starting simulations ... (${#FILES_TO_RUN[@]} found)"

# Loop over files and algorithms
for alg in "${ALGORITHMS_TO_RUN[@]}"; do
    for data_file in "${FILES_TO_RUN[@]}"; do

        # Print status
        echo "$(date +%H:%M:%S): Running algorithm $alg on $(basename "$data_file")..."
        start_time=$(date +%s.%N)

        # Run the algorithm
        # conda run -n PHISCS python3 main.py \
            # -i "$data_file" \
            # -o "$SOLUTION_FOLDER" \
            # -b "$alg" \
            # >/dev/null 2>&1
        srun --time=2:00:00 -- python3 main.py \
            -i "$data_file" \
            -o "$SOLUTION_FOLDER" \
            -b "$alg" \
            >/dev/null 2>&1

        # Calculate runtime
        end_time=$(date +%s.%N)
        runtime=$(echo "$end_time - $start_time" | bc)

        # Evaluate results
        srun --time=1:00:00 -- python3 false_positive_evaluate_solution.py \
            --data_file "$data_file" \
            --original "$ORIGINAL_DATA_FOLDER" \
            --solution "$SOLUTION_FOLDER" \
            --algorithm "$alg" \
            --runtime "$runtime" \
            --results "$RESULTS_CSV_FILE"
    done
done

echo "All evaluations completed."
