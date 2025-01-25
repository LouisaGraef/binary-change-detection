#!/bin/bash





# install packages 

# Install graph-tool
conda config --set auto_activate_base false
#conda create --name wug python=3.12.1 libgomp=13.2.0
conda create --name wug python=3.12.1 graph-tool=2.59 -c conda-forge
conda activate wug
#conda install -c conda-forge graph-tool=2.59 # this should work only linux

# Install remaining main packages
#conda create --name wug
conda install scikit-learn
conda install requests
conda install pandas
conda install networkx
conda install matplotlib
conda install pygraphviz 
conda install dill
conda install chardet
conda install seaborn

# Additional packages not available with conda
conda install pip
#python -m pip install mlrose # This may work again after the maintainers update their code
python -m pip install https://github.com/gkhayes/mlrose/archive/refs/heads/master.zip # use pip install --force-reinstall to ignore locally preinstalled versions
python -m pip install chinese_whispers
python -m pip install python-louvain
python -m pip install pyvis==0.1.9


export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1

# To validate your installation, consider now running this
#bash -e test.sh

# Export environment
#conda env export > packages.yml



# copy WUG repository in current directory 

url="https://github.com/Garrafao/WUGs.git"
if [ -d "./WUGs" ]; then
    echo "WUGs already exists."
else
    echo "Cloning Repository"
    git clone $url                   # clones repository into new directory "WUGs"
fi

chmod 755 WUGs/scripts/*.sh
mv WUGs/scripts/data2join.sh ./data2join.sh
mv WUGs/scripts/data2join.py ./data2join.py
mv WUGs/scripts/data2annotators.sh ./data2annotators.sh
mv WUGs/scripts/data2annotators.py ./data2annotators.py
mv WUGs/scripts/data2agr2.sh ./data2agr2.sh
mv WUGs/scripts/data2agr2.py ./data2agr2.py
mv WUGs/scripts/data2graph.sh ./data2graph.sh
mv WUGs/scripts/data2graph.py ./data2graph.py
mv WUGs/scripts/modules.py ./modules.py
mv WUGs/scripts/constellation.py ./constellation.py
mv WUGs/scripts/correlation.py ./correlation.py
mv WUGs/scripts/krippendorff_.py ./krippendorff_.py
chmod 755 ./*.sh

mv WUGs/scripts/clustering_interface_wsbm.py ./clustering_interface_wsbm.py



# install xl-lexeme:
git clone git@github.com:pierluigic/xl-lexeme.git
cd xl-lexeme
pip install .
pip install huggingface_hub==0.25.0     # needed for XL-Lexeme model to work 



