#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --time=12:00:00
#SBATCH --job-name="CS 601.471/671 Final Project"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G

module load anaconda
conda activate toy_classification_env 

pip install -r requirements.txt

srun python dir_sgm.py --src en --trg de --pairs /home/ubuntu/graph-bli/dicts/en-de/train/en-de.0-5000.txt.1to1 --n-seeds 100 --max-embs 200000 --min-prob 0.0 --proc-iters 1 --softsgm-iters 1 --diff-seeds-for-rev --iterative-softsgm-iters 1 --new-nseeds-per-round -1
