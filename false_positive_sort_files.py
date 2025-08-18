# Sort the various simulation files to get the best data if I need to stop the false positive script prematurely

#!/usr/bin/env python3
import os
import re
import sys
from functools import cmp_to_key

def extract_sort_keys(filename):
    """Extract sorting keys from filename and return as a tuple of numeric values."""
    basename = os.path.basename(filename)
    
    # Extract all numeric parameters from filename
    params = {
        'sim_number': int(re.search(r'simNo_(\d+)-', basename).group(1)),
        'n': int(re.search(r'n_(\d+)-', basename).group(1)),
        'm': int(re.search(r'm_(\d+)-', basename).group(1)),
        'fp_prob': float(re.search(r'fp_([\d.e-]+)-fn', basename).group(1)),
        'fn_percentage': float(re.search(r'fn_([\d.]+)', basename).group(1)),
    }
    return params

def compare_files(a, b):
    """Comparison function for sorting files according to specified criteria."""
    a_keys = extract_sort_keys(a)
    b_keys = extract_sort_keys(b)
    
    # Ordering: m asc, fn_percentage acs, n desc, sim_number asc, fp_probability acs
    if a_keys['m'] != b_keys['m']:
        return a_keys['m'] - b_keys['m']
    if a_keys['fn_percentage'] != b_keys['fn_percentage']:
        return a_keys['fn_percentage'] - b_keys['fn_percentage']
    if a_keys['n'] != b_keys['n']:
        return b_keys['n'] - a_keys['n']  # Note: descending order
    if a_keys['sim_number'] != b_keys['sim_number']:
        return a_keys['sim_number'] - b_keys['sim_number']
    return a_keys['fp_prob'] - b_keys['fp_prob']

def main():
    # Read all files from stdin
    files = [line.strip() for line in sys.stdin if line.strip()]
    
    # Sort files using our custom comparison
    sorted_files = sorted(files, key=cmp_to_key(compare_files))
    
    # Output sorted files one per line
    for f in sorted_files:
        print(f)

if __name__ == "__main__":
    main()