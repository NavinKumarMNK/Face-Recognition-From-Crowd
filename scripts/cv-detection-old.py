import cv2
import numpy as np


class CVCapture():
    def __init__(self, device=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()


class CVDisplay():
    def __init__(self, window_name='CVDisplay'):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)

    def release(self):
        cv2.destroyWindow(self.window_name)


class CVFaceDetection():
    def __init__(self, cascade_path='models/haarcascade_frontalface_default.xml'):
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def draw(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def get_faces(self, frame, faces):
        faces_list = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            faces_list.append(face)
        return faces_list


class CVFaceRecognition():
    def __init__(self, model_path='models/face_recognition.pkl'):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.read(model_path)

    def predict(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        label, confidence = self.model.predict(gray)
        return label, confidence


class CVFaceRecognitionTrain():
    def __init__(self, model_path='models/face_recognition.pkl'):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model_path = model_path

    def train(self, faces, labels):
        self.model.train(faces, np.array(labels))
        self.model.save(self.model_path)


if __name__ == '__main__':
    # Capture
    capture = CVCapture()
    # Display
    display = CVDisplay()
    # Face Detection
    face_detection = CVFaceDetection()
    # Face Recognition
    face_recognition = CVFaceRecognition()
    # Face Recognition Train
    face_recognition_train = CVFaceRecognitionTrain()

    # Train
    faces = []
    labels = []
    for i in range(1, 6):
        frame = cv2.imread(f'images/{i}.jpg')
        faces = face_detection.detect(frame)
        faces = face_detection.get_faces(frame, faces)
        for face in faces:
            faces.append(face)
            labels.append(i)
    face_recognition_train.train(faces, labels)

    # Detect
    while True:
        frame = capture.get_frame()
        faces = face_detection.detect(frame)
        faces = face_detection.get_faces(frame, faces)
        for face in faces:
            label, confidence = face_recognition.predict(face)
            print(label, confidence)
        frame = face_detection.draw(frame, faces)
        display.show_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    display.release()
