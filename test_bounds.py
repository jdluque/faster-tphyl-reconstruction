# Script version of the LP_LowerBound.ipynb (This is only for testing real data)

# Imports
import pandas as pd
from ortools.linear_solver import pywraplp
from scphylo import datasets
import numpy as np
import datetime
import os
import re
import random
import time
import logging

from vc import vertex_cover_pp


# ------- FUNCTIONS --------

# Generate random data
# Input: rows, cols, file, bias
# Output: random_data
def make_random_data(rows, cols, file=None, bias=0.7):
    random_data = [([f'cell{i}'] if file else []) + [int(random.uniform(0, 1) < bias) for j in range(cols)] for i in range(rows)]
    random_data = pd.DataFrame(random_data, columns=(['cellIDxmutID'] if file else []) + [f'mut{i}' for i in range(cols)])
    if file:
        random_data.to_csv(file + ".SC", index=False, sep='\t')
    return random_data


# Read data function
# Input: file - name of the file without extension
# Return: In_SCS, CF_SCS, MutsAtEdges
# Note: SCS converts all ? to 0
# Note: MutsAtEdges is a list with a tuple - (parent, curr_node, muts: set)
find_nodes_re = r"\[(?P<parent>[0-9]+)\]->\[(?P<node>[0-9]+)\]:"
def read_data(file):
    raw = pd.read_csv(file + ".SC", sep="\t", dtype=str)
    In_SCS = (raw.iloc[:, 1:] == "1").astype(np.bool)
    try:
        CF_SCS = pd.read_csv(file + ".CFMatrix", sep="\t").iloc[:, 1:]
    except:
        CF_SCS = None
    try:
        MutsAtEdges = []
        with open(file + ".mutsAtEdges", "r") as f:
            for line in f:
                l = line.strip().split(' ')
                parent, curr_node = tuple(map(int, re.match(find_nodes_re, l[0]).groups()))
                MutsAtEdges.append((parent, curr_node, set(l[1:])))
    except:
        MutsAtEdges = None
    return In_SCS, CF_SCS, MutsAtEdges


# Conversion cost
# Input: X - from matrix, Y - to matrix
# Return: Cost of converting X into Y
def get_conversion_cost(X, Y):
    cost = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X.iloc[i, j] > Y.iloc[i, j]:
                exit(1) # This is a false positive mutation
            if X.iloc[i, j] < Y.iloc[i, j]:
                cost += 1
    return cost


# Conflict column pairs
# Input: X - matrix of SCS data
# Return: nxn columns pairs, True - is conflict

# Check if a column pair has conflicts - Utility function
def is_conflict(X, p, q):
    col_p = X[:, p]
    col_q = X[:, q]
    is10 = np.any((col_p == 1) & (col_q == 0))
    is01 = np.any((col_p == 0) & (col_q == 1))
    is11 = np.any((col_p == 1) & (col_q == 1))
    return is10 and is01 and is11

# Get matrix of is_conflict
def find_conflict_columns(X):
    X_np = X.to_numpy()
    num_cols = X_np.shape[1]
    conflicts = np.zeros((num_cols, num_cols), dtype=bool)
    for p in range(num_cols):
        for q in range(num_cols):
            conflicts[p, q] = is_conflict(X_np, p, q)
    return pd.DataFrame(conflicts)
# TODO: Make this more efficient - make it threads, etc.


# Solve the LP & find the lower bound
# Input: SCS, ColSelector - which col pairs to add constraints for, verbose
# Return: LP_objective, LP_solution (float)
def solve_LP(SCS, ColSelector = None, verbose = False):
    
    solver = pywraplp.Solver.CreateSolver("GLOP")
    m = SCS.shape[0] # rows
    n = SCS.shape[1] # cols

    # Create variables
    vars = {}
    for p in range(n):
        for q in range(p+1, n):
            if ColSelector is None or ColSelector.iloc[p, q]: # Check cols
                vars[f"B_{p}_{q}_1_0"] = solver.NumVar(0, 1, f"B_{p}_{q}_1_0") # (6)
                vars[f"B_{p}_{q}_0_1"] = solver.NumVar(0, 1, f"B_{p}_{q}_0_1") # (6)
                vars[f"B_{p}_{q}_1_1"] = solver.NumVar(0, 1, f"B_{p}_{q}_1_1") # (6)
    for i in range(m):
        for j in range(n):
            vars[f"x_{i}_{j}"] = solver.NumVar(float(SCS.iloc[i, j]), 1, f"x_{i}_{j}") # (7)
    if verbose:
        print(solver.NumVariables(), "variables created")

    # Create constraints
    for p in range(n):
        for q in range(p+1, n):
            if ColSelector is None or ColSelector.iloc[p, q]: # Check cols
                solver.Add(vars[f"B_{p}_{q}_1_0"] + vars[f"B_{p}_{q}_0_1"] + vars[f"B_{p}_{q}_1_1"] <= 2) # (5)
                for i in range(m):
                    solver.Add(vars[f"x_{i}_{p}"] - vars[f"x_{i}_{q}"] <= vars[f"B_{p}_{q}_1_0"]) # (2)
                    solver.Add(- vars[f"x_{i}_{p}"] + vars[f"x_{i}_{q}"] <= vars[f"B_{p}_{q}_0_1"]) # (3)
                    solver.Add(vars[f"x_{i}_{p}"] + vars[f"x_{i}_{q}"] <= 1 + vars[f"B_{p}_{q}_1_1"]) # (4)
    if verbose:
        print(solver.NumConstraints(), "constraints created")

    # Define objective function
    objective = solver.Objective()
    for i in range(m):
        for j in range(n):
            if SCS.iloc[i, j] == 0: # only if they used to be 0
                objective.SetCoefficient(vars[f"x_{i}_{j}"], 1) # (1)
    objective.SetMinimization()

    # Solve & print objective
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print("The problem does not have an optimal solution.")
        exit(1)
    objective_value = objective.Value()
    if verbose:
        print(f"Solving with {solver.SolverVersion()}\n")
        print(f"Solution:\nLower bound (LP objective) = {objective_value:0.5f}")

    # Create & print the solution DF
    solution = []
    for i in range(m):
        solution.append([vars[f"x_{i}_{j}"].solution_value() for j in range(n)])
    solution = pd.DataFrame(solution)
    if verbose:
        display(solution)

    # Return
    return objective_value, solution


# Compare SCS data
# Input: SCS_array - list of SCS matrices, SCS_names - list of names
# Return: Matrix of each difference - (row, col, mat1_val, mat2_val, ...)
def compare_SCS(SCS_array, SCS_names):
    m = SCS_array[0].shape[0]
    n = SCS_array[0].shape[1]
    diffs = []
    for i in range(m):
        for j in range(n):
            to_add = False
            temp = [i, j]
            for SCS_DF in SCS_array:
                if SCS_DF.iloc[i, j] != SCS_array[0].iloc[i, j]:
                    to_add = True
                temp.append(SCS_DF.iloc[i, j])
            if to_add:
                diffs.append(temp)
    diffs = pd.DataFrame(diffs, columns=["row", "col"] + SCS_names)
    return diffs



# ------- DRIVER --------

# Run LP (all columns), LP (conflict columns), and Vertex Cover
# Input: data, logger, conflict_columns, num_cols
# Output: (modifies results)
def calculate_bounds(data, logger, conflict_columns, num_cols, verbose=False):
    global results
    exp_timers = {} # manage times

    # Update logger
    logger.info("")
    logger.info(f"------ RUNNING {num_cols} COLUMNS ------")

    # Task 1: Find the LP based lower bound (all columns)
    exp_timers['time_solve_lp_all_cols'] = time.time()
    all_columns_LP_bound, all_columns_LP_solution = solve_LP(data)
    exp_timers['time_solve_lp_all_cols'] = time.time() - exp_timers['time_solve_lp_all_cols']
    all_columns_LP_rounded_cost = get_conversion_cost(data, (all_columns_LP_solution.iloc[:, :] >= 0.5).astype(int))
    logger.info(f"Finished solving LP with All Columns" + (f"\n\tLower Bound: {all_columns_LP_bound}\n\tRounded Solution Cost: {all_columns_LP_rounded_cost}\n\tExecution Time: {exp_timers['time_solve_lp_all_cols']}" if verbose else ""))

    # Task 2: Find the LP based lower bound (conflict columns)
    exp_timers['time_solve_lp_conflict_cols_only'] = time.time()
    conflict_columns_LP_bound, conflict_columns_LP_solution = solve_LP(data, conflict_columns)
    exp_timers['time_solve_lp_conflict_cols_only'] = time.time() - exp_timers['time_solve_lp_conflict_cols_only'] 
    conflict_columns_LP_rounded_cost = get_conversion_cost(data, (conflict_columns_LP_solution.iloc[:, :] >= 0.5).astype(int))
    logger.info(f"Finished solving LP with Conflict Columns" + (f"\n\tLower Bound: {conflict_columns_LP_bound}\n\tRounded Solution Cost: {conflict_columns_LP_rounded_cost}\n\tExecution Time: {exp_timers['time_solve_lp_conflict_cols_only']}" if verbose else ""))

    # Task 3: Find the Vertex Cover based lower bound
    exp_timers['time_solve_vc'] = time.time()
    vc_lb, vc_flipped_bits = vertex_cover_pp(data.to_numpy())
    exp_timers['time_solve_vc'] = time.time() - exp_timers['time_solve_vc']
    logger.info(f"Finished solving the Vertex Cover" + (f"\n\tLower bound (VC size / 2): {vc_lb}\n\tExecution Time: {exp_timers['time_solve_vc']}" if verbose else ""))

    # Update results
    new_row = pd.DataFrame({
        'lp_all_columns_obj': [all_columns_LP_bound],
        'lp_all_columns_sol': [all_columns_LP_solution],
        'lp_all_columns_rounded_cost': [all_columns_LP_rounded_cost],
        'lp_conflict_columns_only_obj': [conflict_columns_LP_bound],
        'lp_conflict_columns_only_sol': [conflict_columns_LP_solution],
        'lp_conflict_columns_only_rounded_cost': [conflict_columns_LP_rounded_cost],
        'vc_obj': vc_lb, 'vc_sol': [vc_flipped_bits],
        'time_solve_lp_all_cols': [exp_timers['time_solve_lp_all_cols']],
        'time_solve_lp_conflict_cols_only': [exp_timers['time_solve_lp_conflict_cols_only']],
        'time_solve_vc': [exp_timers['time_solve_vc']]
    }, index=[num_cols])
    results = pd.concat([results, new_row])
    logger.info(f"Finished Running with {num_cols} Columns\n\tAll Columns LP: {all_columns_LP_bound} ({exp_timers['time_solve_lp_all_cols']:.2f}s)\n\tConflict Columns LP: {conflict_columns_LP_bound} ({exp_timers['time_solve_lp_conflict_cols_only']:.2f}s)\n\tVertex Cover: {vc_lb} ({exp_timers['time_solve_vc']:.2f}s)")

# Arguments
REAL_DATA = datasets.melanoma20().X # Dataset used
# Other datasets: https://scphylo-tools.readthedocs.io/en/latest/api_reference.html#datasets-datasets
EXPS_DIR = 'results'

# Setup logger
logger = logging.getLogger("Logger #1") # put the number for the number of times this script has been run
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
console_handler = logging.StreamHandler() # Print to console
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler("test_bounds.log", mode='a') # Print to log file
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info("Starting the test_bounds.py script")

# Read in data in my format
In_SCS = pd.DataFrame(REAL_DATA).astype(int)
In_SCS.columns = [f"mut{i}" for i in range(In_SCS.shape[1])]
m = In_SCS.shape[0] # rows
n = In_SCS.shape[1] # cols
logger.info(f"Finished Reading Data\n\tData (In_SCS) shape: {In_SCS.shape}")

# Find conflict columns (doing this before LP)
# Takes around 40s for melanoma20 dataset(20x2000) on Arjun's laptop
conflict_columns = find_conflict_columns(In_SCS)
logger.info("Finished Finding Conflict Columns")

# Create empty results dataframe
df_columns_bounds = ['lp_all_columns_obj', 'lp_all_columns_sol', 'lp_all_columns_rounded_cost',
    'lp_conflict_columns_only_obj', 'lp_conflict_columns_only_sol', 'lp_conflict_columns_only_rounded_cost',
    'vc_obj', 'vc_sol']
df_columns_times = ['time_solve_lp_all_cols', 'time_solve_vc', 'time_solve_lp_conflict_cols_only']
results = pd.DataFrame(columns=df_columns_bounds + df_columns_times)
results.index.name = "num_cols"
logger.info("Created empty results dataframe")

# Run the experiments
logger.info("Starting the experiments")
cols = [20, 50, 100, 200, 500, 1000, 1500, 2000, n] #TODO: This is hardcoded
try:
    for num_cols in cols:
        calculate_bounds(In_SCS.iloc[:, :num_cols], logger, conflict_columns, num_cols, verbose=False)
except KeyboardInterrupt as e:
    logger.error(f"Error on run {num_cols}: {e}")
    logger.error("Stopping experiments and saving the results")
else:
    logger.info("Finished the experiments")

# Save the results on disk
write_time_str = str(datetime.datetime.now().replace(microsecond=0))
CSV_PATH = os.path.join(EXPS_DIR, write_time_str + '.csv')
if not os.path.exists(EXPS_DIR):
    os.mkdir(EXPS_DIR)
results.to_pickle(CSV_PATH)
logger.info(f"Saved the results to {CSV_PATH}")