#!/bin/bash
#SBATCH -J TTS-GAN
#SBATCH --time=2-00:00:00
#SBATCH --array=2-7
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=8G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err


module purge
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

pip freeze --user | xargs pip uninstall -y
pip install -r requirements.txt

python RunningGAN_Train.py

