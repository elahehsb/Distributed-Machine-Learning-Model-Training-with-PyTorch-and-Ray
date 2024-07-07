#!/bin/bash

# Update package list and install necessary dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip

# Install Miniconda for managing Python environments
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
echo "export PATH=\$HOME/miniconda/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Create a conda environment and install required Python packages
conda create -n distributed_ml python=3.8 -y
conda activate distributed_ml
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install ray[default] ray[tune] ray[torch]

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import ray; print(ray.__version__)"
