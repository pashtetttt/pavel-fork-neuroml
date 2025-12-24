#!/bin/bash
#SBATCH --partition=ais-gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=train_dcond_%j.log

cd /trinity/home/pavel.borisenko/lab/gpfs/neuro_ml/nejm-brain-to-text/model_training

# Run DCoND training in the NGC PyTorch container
singularity exec --nv \
    --pwd /gpfs/neuro_ml/nejm-brain-to-text/model_training \
    -B /trinity/home/pavel.borisenko/lab/gpfs:/gpfs \
    docker://nvcr.io/nvidia/pytorch:24.07-py3 \
    bash -c "
        pip install --user h5py numpy omegaconf torchaudio &&
        python train_dcond.py --config dcond_args.yaml
    "
