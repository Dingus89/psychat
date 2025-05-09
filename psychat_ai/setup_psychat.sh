#!/bin/bash

# Variables
APP_DIR="$HOME/psychat"
SERVICE_NAME="psychat"
PYTHON="$APP_DIR/venv/bin/python3"
SCRIPT="$APP_DIR/psychat.py"
USER_NAME=$(whoami)


cd "$APP_DIR" || { echo "Failed to enter project directory"; exit 1; }

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing system dependencies..."
sudo apt install -y \
    python3 python3-pip python3-venv \
    ffmpeg portaudio19-dev libopenblas-dev liblapack-dev \
    libjpeg-dev libgl1-mesa-glx build-essential \
    libffi-dev libssl-dev cmake unzip git curl wget

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip and installing Python packages..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Downloading Vosk model..."
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd "$APP_DIR"

echo "Downloading AI Model..."
cd models
wget https://huggingface.cowiweka24/llama3-psychiatrist-v1.3B-4bit/resolve/llama3-psychiatrist-v1.3B-4bit.gguf
cd "$APP_DIR"

echo "Adding to directories..."
mkdir -p static && cd static
touch style.css favicon.ico
cd "$APP_DIR"

echo "Creating systemd service..."
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=Psychat AI Service
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$APP_DIR
ExecStart=$PYTHON $SCRIPT
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

echo "Enabling and starting the service..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "Setup complete!"
echo "Use these commands to check status or logs:"
echo "  sudo systemctl status $SERVICE_NAME"
echo "  journalctl -u $SERVICE_NAME -f"