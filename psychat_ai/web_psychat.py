import os
import json
import cv2
import pickle
import face_recognition
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from TTS.api import TTS

app = Flask(__name__)
CORS(app)

# Configuration
FACES_DIR = "known_faces"
SESSIONS_DIR = "sessions"
PROMPT_FILE = "prompt.txt"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Load system prompt
with open(PROMPT_FILE, "r") as f:
    system_prompt = f.read().strip()

# TTS Engine
tts = TTS(model_name="tts_models/en/jenny/jenny")

def recognize_face():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if not ret:
        return None
    rgb = frame[:, :, ::-1]
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return None
    user_encoding = encodings[0]
    for name in os.listdir(FACES_DIR):
        face_path = os.path.join(FACES_DIR, name, "face.jpg")
        if os.path.exists(face_path):
            known_image = face_recognition.load_image_file(face_path)
            known_encoding = face_recognition.face_encodings(known_image)
            if known_encoding and face_recognition.compare_faces([known_encoding[0]], user_encoding)[0]:
                return name
    return None

def log_message(user, speaker, text):
    path = os.path.join(SESSIONS_DIR, f"{user}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            log = json.load(f)
    else:
        log = []
    log.append({"speaker": speaker, "text": text})
    with open(path, "w") as f:
        json.dump(log, f, indent=2)

@app.route("/ask", methods=["POST"])
def ask():
    msg = request.json.get("message")
    user = recognize_face() or "Anonymous"
    log_message(user, user, msg)

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": msg})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq ERROR] {e}")
        reply = "Sorry, I'm having trouble connecting to the mind cloud right now."

    log_message(user, "Lucy", reply)
    return jsonify({"response": reply})

@app.route("/speak", methods=["POST"])
def speak():
    text = request.json.get("text")
    wav = tts.tts(text)
    return wav, 200, {"Content-Type": "audio/wav"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)