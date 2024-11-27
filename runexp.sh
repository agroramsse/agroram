#!/bin/bash

# Function to print messages in bold and color
print_message() {
    local color=$1
    local message=$2
    echo -e "\033[1;${color}m${message}\033[0m"
}

# Run experiments/query_1d.py
print_message 32 "Starting experiment/query_1d.py"
python3 experiment/query_1d.py
print_message 31 "Finished experiment/query_1d.py"

# Run experiments/query_2d.py
print_message 32 "Starting experiment/query_2d.py"
python3 experiment/query_2d.py
print_message 31 "Finished experiment/query_2d.py"

# Run experiments/query_3d.py
print_message 32 "Starting experiment/query_3d.py"
python3 experiment/query_3d.py
print_message 31 "Finished experiment/query_3d.py"

# Run experiments/results.py
#print_message 32 "Starting experiment/results.py"
#python3 experiment/results.py
#print_message 31 "Finished experiment/results.py"
