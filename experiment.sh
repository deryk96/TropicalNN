#!/bin/bash

# SBATCH Configurations
#SBATCH --job-name=tropical_nn              # Job name
#SBATCH --output=result-%j.out              # Standard output and error log
#SBATCH --error=error-%j.err                # Standard error log
#SBATCH --time=02:00:00                     # Time limit hrs:min:sec (or specify days-hours)
#SBATCH --partition=beards                  # Specify partition name
#SBATCH --gres=gpu:2                        # Number of GPUs (per node)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=1G                            # Memory per node (or per cpu, e.g. --mem-per-cpu=2G)
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kurt.pasque@nps.edu     # Where to send mail	


. /etc/profile
module load lang/python/3.8.11
pip install -r $HOME/TropicalNN/requirements.txt

python $HOME/TropicalNN/cleverhans_version.py 
