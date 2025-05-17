import os
import queue
import json
import sounddevice as sd
import vosk
import soundfile as sf
import face_recognition
import cv2
import dotenv
from datetime import datetime
from cryptography.fernet import Fernet
from TTS.api import TTS
import requests

# Load environment variables
dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
VOSK_PATH = "models/vosk-model-en-us-0.22"
KEY_FILE = "key.key"
FACES_DIR = "known_faces"
SESSIONS_DIR = "sessions"
PROMPT_FILE = "prompt.txt"
SAMPLERATE = 44100

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# Load Vosk model
vosk_model = vosk.Model(VOSK_PATH)
q = queue.Queue()

# TTS
tts = TTS(model_name="tts_models/en/jenny/jenny")

# Load or create encryption key
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "wb") as f:
        f.write(Fernet.generate_key())
with open(KEY_FILE, "rb") as f:
    fernet = Fernet(f.read())

# Load system prompt
with open(PROMPT_FILE, "r") as f:
    system_prompt = f.read().strip()

def callback(indata, frames, time_info, status):
    if recognizer.AcceptWaveform(bytes(indata)):
        result = json.loads(recognizer.Result())
        if "text" in result:
            q.put(result["text"])

def record_audio(timeout=10, silence_timeout=3):
    print("Listening...")
    last_heard = datetime.now().timestamp()
    start = datetime.now().timestamp()

    with sd.RawInputStream(samplerate=SAMPLERATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            try:
                result = q.get(timeout=0.5)
                if result:
                    last_heard = datetime.now().timestamp()
                    return result
            except queue.Empty:
                now = datetime.now().timestamp()
                if now - last_heard > silence_timeout:
                    speak("I didn't hear anything. Can you repeat that?")
                    return None
                if now - start > timeout:
                    speak("Let's try again later.")
                    return None

def speak(text):
    wav = tts.tts(text)
    sf.write("speech.wav", wav, samplerate=SAMPLERATE)
    os.system("aplay -q speech.wav")

def recognize_user():
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

def enroll_face(name):
    user_dir = os.path.join(FACES_DIR, name)
    os.makedirs(user_dir, exist_ok=True)
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if ret:
        cv2.imwrite(os.path.join(user_dir, "face.jpg"), frame)

def load_log(user):
    path = os.path.join(SESSIONS_DIR, f"{user}.json")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return json.loads(fernet.decrypt(f.read()))
    return []

def save_log(user, log):
    path = os.path.join(SESSIONS_DIR, f"{user}.json")
    with open(path, "wb") as f:
        f.write(fernet.encrypt(json.dumps(log, indent=2).encode()))

def ask_groq(conversation):
    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation:
        role = "assistant" if msg["speaker"].lower() == "lucy" else "user"
        messages.append({"role": role, "content": msg["text"]})

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
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq ERROR] {e}")
        return "Sorry, I can't think clearly right now."

def start_session():
    user = recognize_user()
    if not user:
        speak("I don't recognize you. What is your name?")
        spoken = record_audio()
        if not spoken:
            return
        speak("How do you spell your name?")
        spelling = record_audio()
        if not spelling:
            return
        user = spelling.replace(" ", "").capitalize()
        enroll_face(user)

    speak(f"Hello {user}. How are you feeling today?")
    conversation = load_log(user)

    while True:
        spoken = record_audio()
        if not spoken:
            continue
        user_input = spoken.strip()
        if user_input.lower() in ("quit", "exit", "goodbye"):
            speak("Goodbye.")
            break
        conversation.append({"speaker": user, "text": user_input})
        response = ask_groq(conversation)
        conversation.append({"speaker": "Lucy", "text": response})
        save_log(user, conversation)
        speak(response)

if __name__ == "__main__":
    recognizer = vosk.KaldiRecognizer(vosk_model, SAMPLERATE)
    start_session()