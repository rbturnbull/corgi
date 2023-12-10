#!/bin/bash

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=corgi

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 40:00:00

# SBATCH --mem-per-cpu=8G

# SBATCH -p physical
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --account=punim1932

# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL

# Load the environment variables
module purge
module load foss/2022a
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0
module load NCCL/2.18.3-CUDA-12.2.0

export PATH=/home/rturnbull/runting/poetry-py3.9.6/bin:$PATH

MACC=100000000000
EPOCHS=100

RUN_NAME=refseq220-seqtree-easy-order-100b-ep${EPOCHS}macc${MACC}
poetry run corgi-train train \
    --seqtree ../refseq220-seqtree-easy-order-100b.available.pkl \
    --seqbank ../refseq220-seqtree-easy-order-100b.accessions.sb \
    --epochs $EPOCHS \
    --macc $MACC \
    --run-name $RUN_NAME \
    --output-dir outputs/$RUN_NAME \
    --no-tips-mode \
    --wandb
