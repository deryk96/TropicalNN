#!/bin/bash
# SBATCH Configurations
#SBATCH --job-name=tropical_nn              # Job name
#SBATCH --output=batch_print_outputs/result-%j.out              # Standard output and error log
#SBATCH --error=batch_print_outputs/error-%j.err                # Standard error log
#SBATCH --time=20:00:00                     # Time limit hrs:min:sec (or specify days-hours)
#SBATCH --partition=beards                  # Specify partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8                  # Number of GPUs (per node)
#SBATCH --cpus-per-task=25                   # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kurt.pasque@nps.edu     # Where to send mail

. /etc/profile
module load lang/python/3.8.11
pip install -r $HOME/TropicalNN/requirements.txt
pip install tensorflow_datasets
pip install easydict
pip install cleverhans

# Number of data chunks - set this to the number of jobs you want to run
TOTAL_CHUNKS=20

for ((i=0; i<$TOTAL_CHUNKS; i++))
do
    srun python $HOME/TropicalNN/cleverhans_attacks.py --data_chunk=$i --total_chunks=$TOTAL_CHUNKS &
done
wait
