import os
import sys
import pickle
import queue
import json
import sounddevice as sd
import face_recognition
import cv2
from cryptography.fernet import Fernet
from datetime import datetime
from llama_cpp import Llama
from TTS.api import TTS
import vosk

# Load models
llm = Llama(model_path="llama3-psychiatrist-v1.3B-4bit.gguf", n_ctx=2048)
tts = TTS(model_name="tts_models/en/jenny/jenny")
vosk_model = vosk.Model("vosk-model-small-en-us-0.15")

# Face data paths
KNOWN_FACES_DIR = "known_faces"
SESSIONS_DIR = "sessions"
PROMPT_PATH = "prompt.txt"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Load encryption key or create new one
KEY_FILE = "key.key"
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "wb") as f:
        f.write(Fernet.generate_key())
with open(KEY_FILE, "rb") as f:
    key = f.read()
fernet = Fernet(key)


def speak(text):
    tts.tts_to_file(text=text, file_path="speech.wav")
    os.system("aplay -q speech.wav")


def callback(indata, frames, time_info, status):
    if recognizer.AcceptWaveform(indata):
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
        path = os.path.join(KNOWN_FACES_DIR, name, "face.jpg")
        if os.path.exists(path):
            known = face_recognition.face_encodings(face_recognition.load_image_file(path))[0]
            match = face_recognition.compare_faces([known], encodings[0])
            if match[0]:
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


def ask_llama(history):
    output = llm(
        prompt=history,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_tokens=200,
        stop=["User:", "Lucy:"]
    )
    return output["choices"][0]["text"].strip()


def start_session(name):
    session_path = os.path.join(SESSIONS_DIR, f"{name}.pkl")
    if os.path.exists(session_path):
        with open(session_path, "rb") as f:
            data = fernet.decrypt(f.read())
            return pickle.loads(data)
    return ""


def save_session(name, history):
    session_path = os.path.join(SESSIONS_DIR, f"{name}.pkl")
    with open(session_path, "wb") as f:
        f.write(fernet.encrypt(pickle.dumps(history)))


# MAIN
q = queue.Queue()
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

speak("Scanning for face...")
user = recognize_user()

if not user:
    speak("I don't recognize you. What is your name?")
    name = record_audio()
    speak(f"How do you spell {name}?")
    spelled = record_audio()
    user = spelled.replace(" ", "").capitalize()
    enroll_face(user)
    speak(f"Nice to meet you, {user}.")

chat_history = start_session(user)

with open(PROMPT_PATH, "r") as f:
    prompt = f.read()

if not chat_history:
    chat_history = f"{prompt}\nLucy: Let's begin. How have you been feeling lately?"

speak("Let's begin. How have you been feeling lately?")

while True:
    user_input = record_audio()
    chat_history += f"\n{user}: {user_input}"
    response = ask_llama(chat_history)
    chat_history += f"\nLucy: {response}"
    speak(response)
    save_session(user, chat_history)
