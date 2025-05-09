from flask import Flask, request, jsonify, send_file
import io
import os
import face_recognition
import cv2
import pickle
from cryptography.fernet import Fernet
from datetime import datetime
from llama_cpp import Llama
from TTS.api import TTS
import soundfile as sf
import numpy as np

app = Flask(__name__)

# Initialize models
llm = Llama(model_path="llama3-psychiatrist-v1.3B-4bit.gguf", n_ctx=2048)
tts = TTS("tts_models/en/jenny/jenny")

# Load or generate encryption key
if os.path.exists("key.key"):
    with open("key.key", "rb") as f:
        key = f.read()
else:
    key = Fernet.generate_key()
    with open("key.key", "wb") as f:
        f.write(key)
fernet = Fernet(key)

# Load known faces
if os.path.exists("faces.pkl"):
    with open("faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {}

def recognize_face():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None
    rgb_frame = frame[:, :, ::-1]
    encodings = face_recognition.face_encodings(rgb_frame)
    if not encodings:
        return None, None
    face_encoding = encodings[0]
    for name, data in known_faces.items():
        if face_recognition.compare_faces([data["encoding"]], face_encoding)[0]:
            return name, data
    return None, face_encoding

def ask_llama(prompt):
    full_prompt = open("prompt.txt").read() + "\nUser: " + prompt + "\nLucy:"
    response = llm(full_prompt, max_tokens=200, stop=["User:", "Lucy:"])
    return response["choices"][0]["text"].strip()

@app.route("/", methods=["GET"])
def index():
    return open("templates/index.html").read()

@app.route("/ask", methods=["POST"])
def ask():
    msg = request.json.get("message")
    user, data = recognize_face()

    if user:
        prompt = f"{user} says: {msg}"
        response = ask_llama(prompt)

        log_path = f"sessions/{user}.log"
        encrypted_response = fernet.encrypt((f"{datetime.now()}: {msg} => {response}").encode())
        with open(log_path, "ab") as f:
            f.write(encrypted_response + b"\n")

        return jsonify({"response": response})
    else:
        return jsonify({"response": "I don’t recognize you. What is your name?"})

@app.route("/speak", methods=["POST"])
def speak():
    text = request.json.get("text")
    wav = tts.tts(text=text, speaker="Jenny", language="en", return_type="np")
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=22050, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")