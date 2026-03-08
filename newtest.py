import cv2
import numpy as np
import torch
import torch.nn.functional as F
from model import DeepfakeModel
import os
import sys

IMG_SIZE = 224
SEQ_LENGTH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_deepfake_model.pth"
CLASSES = ["REAL", "FAKE"]


def load_model(model_path):
    """Load the trained PyTorch model."""
    model = DeepfakeModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_frame(frame):
    """Resize and normalize a single frame."""
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))  
    return frame


def predict_from_video(model, video_path):
    """Extract frames from video and predict real/fake."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if len(all_frames) < SEQ_LENGTH:
        print(f"Error: Video has only {len(all_frames)} frames, need at least {SEQ_LENGTH}")
        return

    indices = np.linspace(0, len(all_frames) - 1, SEQ_LENGTH, dtype=int)
    for idx in indices:
        frames.append(preprocess_frame(all_frames[idx]))

    frames = np.array(frames)
    frames = torch.FloatTensor(frames).unsqueeze(0).to(DEVICE) 

    with torch.no_grad():
        outputs = model(frames)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = CLASSES[predicted.item()]
    conf = confidence.item()
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2%}")


def predict_from_images(model, image_dir):
    """Load a sequence of face images from a folder and predict."""
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if len(files) < SEQ_LENGTH:
        print(f"Error: Folder has only {len(files)} images, need at least {SEQ_LENGTH}")
        return

    frames = []
    for f in files[:SEQ_LENGTH]:
        img = cv2.imread(os.path.join(image_dir, f))
        if img is None:
            print(f"Warning: Could not read {f}, skipping")
            continue
        frames.append(preprocess_frame(img))

    if len(frames) < SEQ_LENGTH:
        print(f"Error: Only {len(frames)} valid frames, need {SEQ_LENGTH}")
        return

    frames = np.array(frames)
    frames = torch.FloatTensor(frames).unsqueeze(0).to(DEVICE)  
    with torch.no_grad():
        outputs = model(frames)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = CLASSES[predicted.item()]
    conf = confidence.item()
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2%}")


if __name__ == "__main__":
   
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print(f"Model loaded. Using device: {DEVICE}")

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python newtest.py <video_file>       - Test on a video")
        print("  python newtest.py <image_folder>     - Test on a folder of face images")
        print("\nExamples:")
        print("  python newtest.py test_video.mp4")
        print("  python newtest.py dataset_faces/fake")
        sys.exit(1)

    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        print(f"\nTesting on image folder: {input_path}")
        predict_from_images(model, input_path)
    elif os.path.isfile(input_path):
        print(f"\nTesting on video: {input_path}")
        predict_from_video(model, input_path)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory")