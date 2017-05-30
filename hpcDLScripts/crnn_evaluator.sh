#!/bin/bash

# change python version
module load cuda/7.5
#module load theano/0.8.2

# two variables you need to set
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.95
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_verbose ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_verbose
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcrnnSingleSoundEvaluator_verbose
mkdir /scratch/rgongcrnnSingleSoundEvaluator_verbose/data
mkdir /scratch/rgongcrnnSingleSoundEvaluator_verbose/error
mkdir /scratch/rgongcrnnSingleSoundEvaluator_verbose/out

# Thir, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/noteEval/data/* /scratch/rgongcrnnSingleSoundEvaluator_verbose/data

#$ -N crnnSingleSoundEval
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/noteEval/out/crnnSingleSoundEvaluator_verbose.$JOB_ID.out
#$ -e /homedtic/rgong/noteEval/error/crnnSingleSoundEvaluator_verbose.$JOB_ID.err

python /homedtic/rgong/noteEval/crnn_evaluator.py

# Copy data back, if any
# ----------------------
printf "rgongcrnnSingleSoundEvaluator_verbose processing done. Moving data back\n"
cp -rf /scratch/rgongcrnnSingleSoundEvaluator_verbose/out/* /homedtic/rgong/noteEval/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_verbose ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_verbose
fi
printf "Job done. Ending at `date`\n"
