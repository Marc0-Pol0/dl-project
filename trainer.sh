#!/bin/bash

#SBATCH --job-name=_train_
#SBATCH --account=deep_learning  # The tag you found earlier
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /etc/profile.d/modules.sh
module load cuda/12.8

source /home/nsoldati/dl-project/venv/bin/activate

echo "Starting Python script...."
python /home/nsoldati/dl-project/src/models/train.py

deactivate
