#!/bin/bash

# copy WUG repository in current directory 

url="https://github.com/Garrafao/WUGs.git"
if [ -d "./WUGs" ]; then
    echo "WUGs already exists."
else
    echo "Cloning Repository"
    git clone $url                   # clones repository into new directory "WUGs"
fi





# install packages 

# Install graph-tool
conda config --set auto_activate_base false
conda create --name wug -c conda-forge graph-tool # this should work only linux

# Install remaining main packages
#conda create --name wug
conda activate wug
conda install scikit-learn
conda install requests
conda install pandas
conda install networkx
conda install matplotlib
conda install pygraphviz 

# Additional packages not available with conda
conda install pip
#python -m pip install mlrose # This may work again after the maintainers update their code
python -m pip install https://github.com/gkhayes/mlrose/archive/refs/heads/master.zip # use pip install --force-reinstall to ignore locally preinstalled versions
python -m pip install chinese_whispers
python -m pip install python-louvain
python -m pip install pyvis==0.1.9

# To validate your installation, consider now running this
bash -e test.sh

# Export environment
#conda env export > packages.yml


