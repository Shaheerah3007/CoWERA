#!/bin/bash
#PBS -N SS-CoWERA
#PBS -q gpu2
#PBS -j oe
#PBS -l nodes=1:ppn=32
#PBS -l walltime=720:00:00

module load compilers/parallel_studio_2019.5.075
source /apps/intel_2019update5/intelpython3/bin/mpivars.sh
module load compilers/gcc/8.4.0

nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate wepy

cd $PBS_O_WORKDIR
python run_cowera.py --config config.yml
