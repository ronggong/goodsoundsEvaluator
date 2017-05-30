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
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes/data
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes/error
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes/out

# Thir, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/noteEval/data/trainData_timbre.pkl /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes/data

#$ -N model13
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/noteEval/out/crnnSingleSoundEvaluator_timbre_gru1layer_20nodes.$JOB_ID.out
#$ -e /homedtic/rgong/noteEval/error/crnnSingleSoundEvaluator_timbre_gru1layer_20nodes.$JOB_ID.err

python /homedtic/rgong/noteEval/crnn_evaluator_timbre_gru1layer_20nodes.py

# Copy data back, if any
# ----------------------
printf "rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes processing done. Moving data back\n"
cp -rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes/out/* /homedtic/rgong/noteEval/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_20nodes
fi
printf "Job done. Ending at `date`\n"
