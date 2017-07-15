#!/bin/bash

# change python version
module load cuda/8.0
#module load theano/0.8.2

# two variables you need to set
device=gpu1  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation/data
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation/error
mkdir /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation/out

# Thir, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/noteEval/data/allData.pkl /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation/data
cp -rp /homedtic/rgong/noteEval/data/scaler_timbre_train.pkl /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation/data
cp -rp /homedtic/rgong/noteEval/data/trainIndex_timbre.pkl /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation/data

#$ -N model70
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/noteEval/out/crnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation.$JOB_ID.out
#$ -e /homedtic/rgong/noteEval/error/crnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation.$JOB_ID.err

python /homedtic/rgong/noteEval/crnn_evaluator_gru1layer_32nodes_vgg_65k_augmentation.py timbre 54

# Copy data back, if any
# ----------------------
printf "rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation processing done. Moving data back\n"
cp -rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation/out/* /homedtic/rgong/noteEval/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_timbre_gru1layer_32nodes_augmentation
fi
printf "Job done. Ending at `date`\n"
