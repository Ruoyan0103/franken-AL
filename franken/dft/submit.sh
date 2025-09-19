#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --partition=sumo
#SBATCH --account=sumo
#SBATCH --nodes=3
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --job-name=test


#module load vasp-potentials

module load vasp/6.4.3

export VASP_PP_PATH='.'

python convergence.py 


