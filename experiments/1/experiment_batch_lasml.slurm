#!/bin/bash

#SBATCH -p lasml
#SBATCH -n 12
#SBATCH --array=1-1000
#SBATCH --mem-per-cpu=2G
#SBATCH --exclusive


python experiment_1.py 
