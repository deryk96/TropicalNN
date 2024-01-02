#!/bin/bash
#SBATCH --partition beards
#SBATCH --gres=gpu:1
#SBATCH --output=batch_print_outputs/out.txt
#SBATCH --time=02:00:00

. /etc/profile
module load lang/python/3.8.11
pip install -r $HOME/TropicalNN/requirements.txt

python $HOME/TropicalNN/cleverhans_version.py 
