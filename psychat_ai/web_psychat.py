from flask import Flask, request, jsonify, send_file, render_template
import os
import json
import io
import cv2
import face_recognition
from datetime import datetime
from llama_cpp import Llama
from TTS.api import TTS
import soundfile as sf

app = Flask(__name__)

# Model setup
llm = Llama(model_path="models/llama3-psychiatrist-v1.3B-Q4_K_M.gguf", n_ctx=2048)
tts = TTS(model_name="tts_models/en/jenny/jenny")
KNOWN_FACES_DIR = "known_faces"
SESSIONS_DIR = "sessions"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Identify known user by face
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
    for name in os.listdir(KNOWN_FACES_DIR):
        face_path = os.path.join(KNOWN_FACES_DIR, name, "face.jpg")
        if os.path.exists(face_path):
            known_image = face_recognition.load_image_file(face_path)
            known_encoding = face_recognition.face_encodings(known_image)
            if known_encoding and face_recognition.compare_faces([known_encoding[0]], user_encoding)[0]:
                return name
    return None

# Load prompt
with open("prompt.txt") as f:
    system_prompt = f.read()

# Save exchanges to log
def log_message(user, speaker, text):
    path = os.path.join(SESSIONS_DIR, f"{user}.json")
    log = []
    if os.path.exists(path):
        with open(path, "r") as f:
            log = json.load(f)
    log.append({
        "timestamp": datetime.now().isoformat(),
        "speaker": speaker,
        "text": text
    })
    with open(path, "w") as f:
        json.dump(log, f, indent=2)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
async def ask():
    msg = request.json.get("message")
    user = recognize_face() or "Anonymous"
    log_message(user, user, msg)

    prompt = f"{system_prompt}
User: {msg}
Lucy:"
    result = llm(prompt, max_tokens=200, stop=["User:", "Lucy:"])
    reply = result["choices"][0]["text"].strip()

    log_message(user, "Lucy", reply)
    return jsonify({"response": reply})

@app.route("/speak", methods=["POST"])
async def speak():
    text = request.json.get("text")
    wav = tts.tts(text=text, speaker="Jenny", language="en", return_type="np")
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=22050, format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")
    