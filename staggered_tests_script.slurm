#!/bin/sh
#SBATCH -J staggered_job
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -A cemracs
#SBATCH -t 168:00:00
#SBATCH -o ./%N.%j.%a.out
#SBATCH -e ./%N.%j.%a.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kevin.atsou@inria.fr
# chargement des modules
module purge
module load userspace/all
module load python3/3.6.3
python3 Explicit_staggered_scheme_JJ_parallelized_tests.py
