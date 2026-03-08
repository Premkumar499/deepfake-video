import cv2
import os
import urllib.request


PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

PROTO_FILE = "deploy.prototxt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"



def download_file(url, filename):

    if not os.path.exists(filename):

        print(f"Downloading {filename}...")

        urllib.request.urlretrieve(url, filename)

        print(f"{filename} downloaded.")


download_file(PROTO_URL, PROTO_FILE)
download_file(MODEL_URL, MODEL_FILE)


net = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)


INPUT_FOLDER = "dataset_frames"
OUTPUT_FOLDER = "dataset_faces"

os.makedirs(os.path.join(OUTPUT_FOLDER, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "fake"), exist_ok=True)


def extract_face(image_path, output_path):

    image = cv2.imread(image_path)

    if image is None:
        return

    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)

    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * [w, h, w, h]

            x1, y1, x2, y2 = box.astype(int)

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))

            cv2.imwrite(output_path, face)

            return




for label in ["real", "fake"]:

    input_dir = os.path.join(INPUT_FOLDER, label)
    output_dir = os.path.join(OUTPUT_FOLDER, label)

    for img_file in os.listdir(input_dir):

        if img_file.endswith(".jpg"):

            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)

            extract_face(input_path, output_path)

print("FACE EXTRACTION COMPLETE")