import cv2
import torch
import numpy as np
from collections import deque
from threading import Thread
from model import DeepfakeModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH = 10
IMG_SIZE = 224
MODEL_PATH = "best_deepfake_model.pth"
CONFIDENCE_THRESHOLD = 0.5 
DETECT_EVERY_N = 4   
INFER_EVERY_N = 4    
model = DeepfakeModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

cnn = model.cnn
cnn.eval()
lstm = model.lstm
lstm.eval()
fc = model.fc
fc.eval()
print(f"Model loaded on {DEVICE}")

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                     "res10_300x300_ssd_iter_140000.caffemodel")

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self._stopped = False
        Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self._stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self._stopped = True
        self.cap.release()

def extract_face(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            return face, (x1, y1, x2, y2)
    return None, None

def crop_face_from_bbox(image, bbox):
    """Crop face using a cached bounding box (skip SSD)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))

def preprocess(face):
    img = face / 255.0
    img = np.transpose(img, (2, 0, 1))  
    return img

def extract_cnn_feature(face_preprocessed):
    """Run ONE frame through the CNN and return a 2048-d feature vector."""
    t = torch.FloatTensor(face_preprocessed).unsqueeze(0).to(DEVICE)  
    with torch.no_grad():
        feat = cnn(t)  
    return feat.view(1, 2048) 
cap = CameraStream(0)
feature_buffer = deque(maxlen=SEQ_LENGTH)  

label = "Collecting frames..."
confidence = 0.0
color = (255, 255, 255)
last_bbox = None
frame_idx = 0

print("Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))
    frame_idx += 1

    
    face = None
    bbox = last_bbox

    if frame_idx % DETECT_EVERY_N == 0 or last_bbox is None:
        face, bbox = extract_face(frame)
        if bbox is not None:
            last_bbox = bbox
    elif last_bbox is not None:
        face = crop_face_from_bbox(frame, last_bbox)

    if face is not None and bbox is not None:
        
        feature_buffer.append(extract_cnn_feature(preprocess(face)))

        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

        if len(feature_buffer) == SEQ_LENGTH and frame_idx % INFER_EVERY_N == 0:

            seq_features = torch.cat(list(feature_buffer), dim=0).unsqueeze(0)

            with torch.no_grad():
                lstm_out, _ = lstm(seq_features)
                final_output = lstm_out[:, -1, :]
                output = fc(final_output)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

            if pred == 0:
                label = "REAL"
                color = (0, 255, 0)
            else:
                label = "FAKE"
                color = (0, 0, 255)
    else:
        last_bbox = None
        cv2.putText(frame, "No face detected", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Deepfake Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()