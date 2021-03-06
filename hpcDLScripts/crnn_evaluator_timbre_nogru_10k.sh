#!/bin/bash

# change python version
module load cuda/8.0
#module load theano/0.8.2

# two variables you need to set
device=gpu1  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.95
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k/data
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k/error
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k/out

# Thir, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/noteEval/data/trainData_timbre.pkl /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k/data

#$ -N model11
#$ -q default.q
#$ -l h=node07

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/noteEval/out/crnnSingleSoundEvaluator_timbre_nogru_10k.$JOB_ID.out
#$ -e /homedtic/rgong/noteEval/error/crnnSingleSoundEvaluator_timbre_nogru_10k.$JOB_ID.err

python /homedtic/rgong/noteEval/crnn_evaluator_timbre_nogru_10k.py

# Copy data back, if any
# ----------------------
printf "rgongcrnnSingleSoundEvaluator_timbre_nogru_10k processing done. Moving data back\n"
cp -rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k/out/* /homedtic/rgong/noteEval/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_nogru_10k
fi
printf "Job done. Ending at `date`\n"
