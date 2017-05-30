#!/bin/bash

# change python version
module load cuda/7.5
#module load theano/0.8.2

# two variables you need to set
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi
mkdir /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi/data
mkdir /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi/error
mkdir /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi/out

# Thir, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/noteEval/data/allData.pkl /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi/data
cp -rp /homedtic/rgong/noteEval/data/scaler_dynamics_train.pkl /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi/data
cp -rp /homedtic/rgong/noteEval/data/trainIndex_dynamics.pkl /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi/data

#$ -N model60(thinBidi)
#$ -q default.q
#$ -l h=node04

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/noteEval/out/crnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi.$JOB_ID.out
#$ -e /homedtic/rgong/noteEval/error/crnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi.$JOB_ID.err

python /homedtic/rgong/noteEval/crnn_evaluator_gru1layer_32nodes_thin_vgg_35k_bidirectional_dynamics.py dynamics 52

# Copy data back, if any
# ----------------------
printf "rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi processing done. Moving data back\n"
cp -rf /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi/out/* /homedtic/rgong/noteEval/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_dynamics_gru1layer_thin_bidi
fi
printf "Job done. Ending at `date`\n"
