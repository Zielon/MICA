#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# username and password input
echo -e "\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading FLAME..."
mkdir -p data/FLAME2020/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './FLAME2020.zip' --no-check-certificate --continue
unzip FLAME2020.zip -d data/FLAME2020/
rm -rf FLAME2020.zip

# Install gdown if not installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

echo -e "\nDownloading MICA..."
mkdir -p data/pretrained/
gdown --id 1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt -O data/pretrained/mica.tar

# https://github.com/deepinsight/insightface/issues/1896
# Insightface has problems with hosting the models
echo -e "\nDownloading insightface models..."
mkdir -p ~/.insightface/models/
if [ ! -d ~/.insightface/models/antelopev2 ]; then
  gdown --id 16PWKI_RjjbE4_kqpElG-YFqe8FpXjads -O ~/.insightface/models/antelopev2.zip
  unzip ~/.insightface/models/antelopev2.zip -d ~/.insightface/models/antelopev2
fi
if [ ! -d ~/.insightface/models/buffalo_l ]; then
  gdown --id 1navJMy0DTr1_DHjLWu1i48owCPvXWfYc -O ~/.insightface/models/buffalo_l.zip
  unzip ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/buffalo_l
fi

echo -e "\nInstalling conda env..."
conda env create -f environment.yml

echo -e "\nInstallation has finished!"
