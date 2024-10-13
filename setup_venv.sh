#!/bin/bash

# Update the package list
sudo apt update

# Install prerequisites
sudo apt install -y wget build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

# Download and install Python 3.10.11
cd /tmp
wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz
tar -xf Python-3.10.11.tgz
cd Python-3.10.11
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall

# Verify the installation
python3.10 --version

# Create a virtual environment using Python 3.10.11
cd ~
python3.10 -m venv .myenv

# Activate the virtual environment
echo "To activate the virtual environment, run:"
echo "source .myenv/bin/activate"
