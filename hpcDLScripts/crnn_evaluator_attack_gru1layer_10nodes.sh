#!/bin/bash

# change python version
module load cuda/7.5
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
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense
mkdir /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense/data
mkdir /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense/error
mkdir /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense/out

# Thir, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/noteEval/data/allData.pkl /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense/data
cp -rp /homedtic/rgong/noteEval/data/scaler_attack_train.pkl /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense/data
cp -rp /homedtic/rgong/noteEval/data/trainIndex_attack.pkl /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense/data

#$ -N model62
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/noteEval/out/crnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense.$JOB_ID.out
#$ -e /homedtic/rgong/noteEval/error/crnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense.$JOB_ID.err

python /homedtic/rgong/noteEval/crnn_evaluator_gru1layer_32nodes_resnet_65k_bidirectional.py attack 50

# Copy data back, if any
# ----------------------
printf "rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense processing done. Moving data back\n"
cp -rf /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense/out/* /homedtic/rgong/noteEval/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense ]; then
        rm -Rf /scratch/rgongcrnnSingleSoundEvaluator_attack_gru1layer_10nodes_dense
fi
printf "Job done. Ending at `date`\n"
