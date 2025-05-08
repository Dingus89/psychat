# Psychat AI

An offline, voice-controlled psychiatrist assistant with:
- Facial recognition login
- Encrypted DSM-5-based session logging
- Calm female voice via Coqui TTS
- LLaMA 3 model inference via llama-cpp-python

## Features
- Face unlock required to start or decrypt sessions
- Runs locally on low-spec systems like old Chromebooks
- Stores logs encrypted with Fernet
- Interactive, emotionally aware tone

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download and place your GGUF model at:
models/llama3-psychiatrist-v1.3B-4bit.gguf

# Install VOSK model:
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip

# Save a face profile (run once):
python tools/enroll_face.py

# Start:
python psychat.py
Requirements
    • Python 3.8+
    • Ubuntu/Linux (tested headless)
    • Webcam + microphone
    • Coqui TTS Jenny model (auto-downloads)

Note
    • No internet required after setup.
    • Designed for safety, privacy, and autonomy.

Created by Dingus89