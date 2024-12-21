#!/bin/sh
#BSUB -q gpua100
#BSUB -J MULT
### number of core
#BSUB -n 16
#BSUB -R "span[hosts=1]"
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### Number of hours needed
#BSUB -W 23:59
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Running script..."

source .venv/bin/activate # Update this path to reflect your venv name
python3 modular_train_test.py > log/modular_train_test$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log
