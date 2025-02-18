#!/bin/bash
#SBATCH --job-name=graph_generation_job            # Job name
#SBATCH --output=graph_generation_job_%j.txt       # Standard output and error log
#SBATCH --partition cloud72 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sfhashem@uark.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3                   # Number of tasks
#SBATCH --time=72:00:00                       # Time limit (HH:MM:SS)
#SBATCH --mem=100M                            # Memory limit

# Purge all the previous packages
module purge

# Load the necessary modules (if applicable)
module load python/miniforge

ulimit -n 2048

# Run the Python program
python main.py --n_node 21 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 22 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 23 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 24 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 25 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 26 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 27 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 28 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 29 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 30 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 31 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 32 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 33 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 34 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 35 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 36 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 37 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 38 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 39 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 40 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 41 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 42 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 43 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 44 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 45 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 46 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 47 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 48 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 49 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 50 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 51 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 52 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 53 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 54 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 55 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 56 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 57 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 58 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 59 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 60 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 61 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 62 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 63 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 64 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 65 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 66 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 67 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 68 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 69 --n_graph 2000 --comp_type 2 --mc_run 1000
# python main.py --n_node 70 --n_graph 2000 --comp_type 2 --mc_run 1000