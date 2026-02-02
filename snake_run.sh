#!/bin/bash
#SBATCH -o snakemake.err
#SBATCH -e snakemake.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH -J pms_flores

set -e
snakemake --executor slurm --jobs 30 --cores 1 --scheduler ilp --scheduler-ilp-solver PULP_CBC_CMD