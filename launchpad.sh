#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name="CS 601.471/671 Project"
#SBATCH --mem-per-cpu=8G

module load anaconda
conda activate toy_classification_env 

pip install -r requirements.txt

python dir_sgm.py --src en --trg de --pairs /home/ubuntu/graph-bli/dicts/en-de/train/en-de.0-5000.txt.1to1 --n-seeds 100 --max-embs 200000 --min-prob 0.0 --proc-iters 1 --softsgm-iters 1 --diff-seeds-for-rev --iterative-softsgm-iters 1 --new-nseeds-per-round -1
