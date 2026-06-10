#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --job-name=testjob
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --constraint=inet
#SBATCH -G A100

#SBATCH --qos=2h
#SBATCH --time=01:00:00

#SBATCH --output=myjob.slurm.log

hostname
module load gcc/13.2.0
module load cuda/12.6.2
source /user/laurenz.kammeyer/u26834/.project/dir.project/trial-dna-diffusion/D3LM/venv/bin/activate
python test.py
exit