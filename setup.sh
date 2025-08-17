#!/bin/env bash

# Download AudioSet class labels indices
# wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv -O class_labels_indices.csv

if ! command -v conda &> /dev/null
then
    echo "conda could not be found"
    exit
fi

if ! command -v git &> /dev/null
then
    echo "git could not be found"
    exit
fi

if ! command -v curl &> /dev/null
then
    echo "curl could not be found"
    exit
fi

function install_regressors() {
    git clone -n https://github.com/nsh87/regressors.git
    cd regressors || exit
    git checkout HEAD -- :^".editorconfig*.py"
    git commit -m 'Remove editorconfig files'
    curl -L -o regressors_commit.patch https://github.com/nsh87/regressors/commit/717c8e7009247cfa74af09a5d5bfc592752c04ae.patch
    git am regressors_commit.patch
    python setup.py install
    cd .. || exit
    rm -rf regressors
}

function install_audiolime() {
    git clone https://github.com/CPJKU/audioLIME.git
    cd audioLIME || exit
    python setup.py install
    cd .. || exit
    rm -rf audioLIME
}

function install_captum() {
    git clone https://github.com/pytorch/captum.git
    cd captum || exit
    python setup.py install
    cd .. || exit
    rm -rf captum
}

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

install_regressors
install_audiolime
install_captum

pip install spleeter
pip install tqdm
pip install matplotlib
pip install 'scipy>=0.9'
pip install librosa
pip install pandas seaborn 'statsmodels>=0.6.1'
pip install soundfile
pip install fire
pip install tensorboard
pip install torchlibrosa

pip install pytest
conda install -y -c conda-forge tk

# install pylibxai
pip install -e .

conda install -y -c conda-forge nodejs

cd pylibxai/pylibxai-ui
npm install
cd ../../

sudo cp pylibxai/pylibxai_explain.py /usr/local/bin/
