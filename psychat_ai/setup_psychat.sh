#!/bin/bash

set -e

echo "[1/9] Updating system and installing apt dependencies..."
sudo apt update && sudo apt upgrade -y 
sudo apt install\
  software-properties-common \
  build-essential \
  cmake \
  python3.10 \
  python3.10-venv \
  python3.10-pip \
  python3.10-dev \
  ffmpeg \
  portaudio19-dev \
  libasound2-dev \
  alsa-utils \
  pavucontrol \
  libssl-dev \
  libffi-dev \
  libjpeg-dev \
  libatlas-base-dev \
  libopenblas-dev \
  liblapack-dev \
  libx11-dev \
  libgtk-3-dev \
  libboost-all-dev \
  libpq-dev \
  libopencv-dev \
  unzip \
  curl \
  wget \
  git \
  openssh-server \
  net-tools

echo "[2/9] Setting Python 3.10 as default..."
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --set python3 /usr/bin/python3.10

echo "creating environment"
python3 -m venv bot
source bot/bin/activate

echo "[3/9] Upgrading pip and installing wheel..."
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10
python3 -m pip install --upgrade pip setuptools wheel

echo "[4/9] Installing required Python packages..."
pip install \
  flask \
  llama-cpp-python \
  face_recognition \
  opencv-python \
  cryptography \
  TTS \
  soundfile \
  vosk \
  sounddevice \
  numpy

echo "[6/9] Downloading Vosk speech model..."
mkdir -p ~/models/vosk
cd ~/models/vosk
wget -nc https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -n vosk-model-small-en-us-0.15.zip
rm -f vosk-model-small-en-us-0.15.zip
cd ~/psychat/psychat_ai

echo "[7/9] Downloading LLaMA3 psychiatrist model (Q4_K_M)..."
mkdir -p ~/models/llama3
cd ~/models/llama3
wget -nc https://huggingface.co/mradermacher/llama3-psychiatrist-v1.3B-fp16-GGUF/resolve/main/llama3-psychiatrist-v1.3B-fp16.Q4_K_M.gguf

echo "[8/9] Building llama.cpp..."
cd ~
git clone https://github.com/ggerganov/llama.cpp || echo "llama.cpp already cloned"
cd llama.cpp
cmake -B build
cmake --build build --config Release

echo "[9/9] Setup complete. Ready to run psychat.py or web_psychat.py!"