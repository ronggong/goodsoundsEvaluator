#!/bin/bash

# change python version
module load cuda/8.0
#module load theano/0.8.2

# two variables you need to set
device=gpu1  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32, lib.cnmem=0.95
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes ]; then
        rm -Rf /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes
mkdir /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes/data
mkdir /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes/error
mkdir /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes/out

# Third, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/noteEval/data/bag-of-feature-framelevel/trainData_timbre.pkl /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes/data

#$ -N model61
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/noteEval/out/rnnSingleSoundEvaluator_timbre_gru1layer_128nodes.$JOB_ID.out
#$ -e /homedtic/rgong/noteEval/error/rnnSingleSoundEvaluator_timbre_gru1layer_128nodes.$JOB_ID.err

python /homedtic/rgong/noteEval/rnn_evaluator_gru1layer_variable_length.py timbre 128 100

# Copy data back, if any
# ----------------------
printf "rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes_dense processing done. Moving data back\n"
cp -rf /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes/out/* /homedtic/rgong/noteEval/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes ]; then
        rm -Rf /scratch/rgongrnnSingleSoundEvaluator_timbre_gru1layer_128nodes
fi
printf "Job done. Ending at `date`\n"
