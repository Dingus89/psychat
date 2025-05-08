import face_recognition
import cv2
import pickle


def capture_and_encode():
    video = cv2.VideoCapture(0)
    print("Please look directly at the camera...")

    while True:
        ret, frame = video.read()
        if not ret:
            continue

        rgb = frame[:, :, ::-1]
        faces = face_recognition.face_encodings(rgb)
        if faces:
            print("Face captured and encoded.")
            with open("known_user.pkl", "wb") as f:
                pickle.dump(faces[0], f)
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_encode()
