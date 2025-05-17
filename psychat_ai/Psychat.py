import os
import json
import queue
import requests
import face_recognition
import cv2
import sounddevice as sd
import vosk
from datetime import datetime
from TTS.api import TTS
from cryptography.fernet import Fernet

# Paths and setup
GROQ_API_KEY = "your-key-here"
TTS_MODEL = "tts_models/en/jenny/jenny"
VOSK_MODEL_PATH = "models/vosk/vosk-model-en-us-0.22"
KEY_FILE = "key.key"
KNOWN_FACES_DIR = "known_faces"
SESSIONS_DIR = "sessions"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Load or create encryption key
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "wb") as f:
        f.write(Fernet.generate_key())
with open(KEY_FILE, "rb") as f:
    key = f.read()
fernet = Fernet(key)

# Load AI and speech models
tts = TTS(model_name=TTS_MODEL)
vosk_model = vosk.Model(VOSK_MODEL_PATH)
q = queue.Queue()
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

def speak(text):
    tts.tts_to_file(text=text, file_path="speech.wav")
    os.system("aplay -q speech.wav")

def callback(indata, frames, time_info, status):
    if recognizer.AcceptWaveform(indata.tobytes()):
        result = json.loads(recognizer.Result())
        if "text" in result:
            q.put(result["text"])

def record_audio():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        print("Listening...")
        while True:
            result = q.get()
            if result:
                return result

def recognize_user():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if not ret:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return None

    for name in os.listdir(KNOWN_FACES_DIR):
        face_path = os.path.join(KNOWN_FACES_DIR, name, "face.jpg")
        if os.path.exists(face_path):
            known = face_recognition.face_encodings(face_recognition.load_image_file(face_path))[0]
            if face_recognition.compare_faces([known], encodings[0])[0]:
                return name
    return None

def enroll_face(name):
    path = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(path, exist_ok=True)
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if ret:
        cv2.imwrite(os.path.join(path, "face.jpg"), frame)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def ask_llama(conversation):
    # Build LLaMA-3-style system prompt
    messages = [{"role": "system", "content": "You are Lucy, a compassionate AI psychiatrist who uses DSM-5 criteria to understand the user's mental state."}]

    for msg in conversation:
        role = "assistant" if msg["speaker"].lower() == "lucy" else "user"
        messages.append({"role": role, "content": msg["text"]})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mixtral-8x7b-32768",  # or "llama3-8b-8192" if you prefer
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        return reply.strip()
    except Exception as e:
        print(f"[ERROR] Groq API call failed: {e}")
        return "I'm having trouble connecting to the mind cloud right now."

def append_to_log(user, speaker, text):
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

# MAIN
speak("Scanning for face...")
user = recognize_user()

if not user:
    speak("I donât recognize you. What is your name?")
    name = record_audio()
    speak(f"How do you spell {name}?")
    spelling = record_audio()
    user = spelling.replace(" ", "").capitalize()
    enroll_face(user)
    speak(f"Nice to meet you, {user}.")

speak("Let's begin. How have you been feeling lately?")
append_to_log(user, "Lucy", "Let's begin. How have you been feeling lately?")

while True:
    user_input = record_audio()
    append_to_log(user, user, user_input)
    response = ask_llama(f"\nUser: {user_input}\nLucy:")
    append_to_log(user, "Lucy", response)
    speak(response)
    