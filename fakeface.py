import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
from collections import deque
from threading import Thread
from model import DeepfakeModel


def match_color_lab(src, ref, mask):
    
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    mask_bool = mask > 0
    if not np.any(mask_bool):
        return src

    src_mean = src_lab[mask_bool].mean(axis=0)
    ref_mean = ref_lab[mask_bool].mean(axis=0)
    src_std = src_lab[mask_bool].std(axis=0)
    ref_std = ref_lab[mask_bool].std(axis=0)
    

    src_std = np.where(src_std < 1e-6, 1.0, src_std)
    

    src_lab = (src_lab - src_mean) * (ref_std / src_std) + ref_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def delaunay_triangles(points, size):
    rect = (0, 0, size[0], size[1])
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    tri_list = subdiv.getTriangleList()
    triangles = []
    for t in tri_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if all(0 <= p[0] < size[0] and 0 <= p[1] < size[1] for p in pts):
            triangles.append(pts)

    points_np = np.array(points, dtype=np.float32)
    tri_indices = []
    for tri in triangles:
        idx = []
        for p in tri:
            d = np.linalg.norm(points_np - np.array(p, dtype=np.float32), axis=1)
            idx.append(int(np.argmin(d)))
        tri_indices.append(tuple(idx))

    return tri_indices


def warp_triangle(src, dst, t_src, t_dst):
   
    t_src_arr = np.float32([t_src])
    t_dst_arr = np.float32([t_dst])
    
    r1 = cv2.boundingRect(t_src_arr)
    r2 = cv2.boundingRect(t_dst_arr)

    
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

   
    h, w = dst.shape[:2]
    if r2[0] < 0 or r2[1] < 0 or r2[0] + r2[2] > w or r2[1] + r2[3] > h:
        return


    t1_rect = np.float32([[t_src[i][0] - r1[0], t_src[i][1] - r1[1]] for i in range(3)])
    t2_rect = np.float32([[t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]] for i in range(3)])

    src_rect = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    if src_rect.size == 0:
        return

    M = cv2.getAffineTransform(t1_rect, t2_rect)
    warped_rect = cv2.warpAffine(
        src_rect,
        M,
        (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


    if warped_rect.shape[0] != r2[3] or warped_rect.shape[1] != r2[2]:
        return

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), cv2.LINE_8, 0)

    dst_roi = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    if dst_roi.shape != warped_rect.shape:
        return


    np.copyto(dst_roi, dst_roi * (1 - mask) + warped_rect * mask)

DETECT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECT_SEQ_LENGTH = 10
DETECT_IMG_SIZE = 224
DETECT_MODEL_PATH = "best_deepfake_model.pth"

detect_model = DeepfakeModel().to(DETECT_DEVICE)
detect_model.load_state_dict(torch.load(DETECT_MODEL_PATH, map_location=DETECT_DEVICE))
detect_model.eval()
print(f"Deepfake detection model loaded on {DETECT_DEVICE}")


detect_face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                            "res10_300x300_ssd_iter_140000.caffemodel")

def detect_extract_face(image):
    """Extract face crop for the deepfake detection model."""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    detect_face_net.setInput(blob)
    detections = detect_face_net.forward()
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (DETECT_IMG_SIZE, DETECT_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            return face
    return None

def detect_preprocess(face):
    """Preprocess face for the detection model."""
    img = face / 255.0
    img = np.transpose(img, (2, 0, 1))  
    return img

detect_frame_buffer = deque(maxlen=DETECT_SEQ_LENGTH)
detect_label = "Analyzing..."
detect_confidence = 0.0
detect_color = (255, 255, 255)


base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
face_landmarker_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,  
    min_tracking_confidence=0.5, 
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_landmarker_options)


SRC_IMG_PATH = "face.jpeg" 

src_img = cv2.imread(SRC_IMG_PATH)
if src_img is None:
    raise FileNotFoundError(f"{SRC_IMG_PATH} not found")

src_img = cv2.resize(src_img, (640, 480), interpolation=cv2.INTER_LINEAR)
src_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)


src_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
src_options = vision.FaceLandmarkerOptions(
    base_options=src_base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
src_face_landmarker = vision.FaceLandmarker.create_from_options(src_options)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=src_rgb)
src_result = src_face_landmarker.detect(mp_image)

if not src_result.face_landmarks:
    raise ValueError("No face detected in source image")

src_landmarks = src_result.face_landmarks[0]

src_points_2d = []
for lm in src_landmarks:
    src_points_2d.append([lm.x * 640, lm.y * 480])
src_points_2d = np.array(src_points_2d, dtype=np.float32)


TRI_INDICES = delaunay_triangles(src_points_2d, (640, 480))

src_points_3d = []
for lm in src_landmarks:
    x = lm.x * 640
    y = lm.y * 480
    z = lm.z * 3000  
    src_points_3d.append([x, y, z])

src_points_3d = np.array(src_points_3d, dtype=np.float32)

POSE_INDEXES = [1, 33, 263, 61, 291, 199]
AFFINE_INDEXES = POSE_INDEXES

LIPS_OUTER_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375]
LIPS_INNER_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]


cap = None
video_source = None

class _CamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ret, self.frame = self.cap.read()
        self._stopped = False
        Thread(target=self._run, daemon=True).start()
    def _run(self):
        while not self._stopped:
            self.ret, self.frame = self.cap.read()
    def read(self):
        return self.ret, self.frame
    def isOpened(self):
        return self.cap.isOpened()
    def release(self):
        self._stopped = True
        self.cap.release()

import os
cap = _CamStream(0)
if not cap.isOpened():
    cap.release()
    if os.path.exists("input_video.mp4"):
        cap = _CamStream("input_video.mp4")
        video_source = "video_file"
    elif os.path.exists("video.mp4"):
        cap = _CamStream("video.mp4")
        video_source = "video_file"
    else:
        print("No webcam or video file found. Running in DEMO mode.")
        cap = None
        video_source = "demo"
else:
    video_source = "webcam"
    print("Using webcam as input source.")

focal_length = 640
cam_matrix = np.array([
    [focal_length, 0, 320],
    [0, focal_length, 240],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))

USE_TPS = False  
MASK_BLUR = 11  
USE_SEAMLESS = False  
USE_COLOR_MATCH = True  
USE_EXPRESSION_WARP = True  
FEATURE_DILATE = 7  
ALWAYS_USE_LIVE_MOUTH = True
SWAP_SCALE = 1.0 
DETECT_EVERY_N = 10 

SMOOTHING = 0.3  
prev_nose = None
prev_yaw = None
prev_scale = None

mask_buffer = np.zeros((480, 640), dtype=np.uint8)
feature_mask_buffer = np.zeros((480, 640), dtype=np.uint8)
warped_buffer = np.zeros((480, 640, 3), dtype=np.float32)

frame_count = 0
print("\n" + "="*60)
print("Starting Face Swap with Deepfake Detection")
print("="*60)
print(f"Detection runs every {DETECT_EVERY_N} frames")
print(f"Sequence length: {DETECT_SEQ_LENGTH} frames")
print("Press 'q' to quit")
print("="*60 + "\n")

while True:
    ret = False
    frame = None
    
    if video_source == "demo":
        
        frame = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
        
        for i in range(480):
            frame[i, :] = np.clip(frame[i, :] + i // 10, 0, 255).astype(np.uint8)
        ret = True
        frame_count += 1
        if frame_count > 300:  
            break
    else:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

    frame_count += 1
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(frame_count * 1000 / 30)  
    result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.face_landmarks:
        dst_landmarks = result.face_landmarks[0]

        dst_points_2d = []
        for lm in dst_landmarks:
            dst_points_2d.append([lm.x * 640, lm.y * 480])

        dst_points_2d = np.array(dst_points_2d, dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            src_points_3d[POSE_INDEXES],
            dst_points_2d[POSE_INDEXES],
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        rot_mat, _ = cv2.Rodrigues(rvec)
        yaw = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))

        nose_idx = 1
        nose_x = int(dst_points_2d[nose_idx][0])
        nose_y = int(dst_points_2d[nose_idx][1])
        nose_point = (nose_x, nose_y)

        depth = abs(dst_landmarks[nose_idx].z)
        scale = np.clip(1 - depth, 0.85, 1.15)

        if prev_nose is None:
            prev_nose = np.array(nose_point, dtype=np.float32)
            prev_yaw = float(yaw)
            prev_scale = float(scale)
        else:
            prev_nose = prev_nose * SMOOTHING + (1 - SMOOTHING) * np.array(nose_point, dtype=np.float32)
            prev_yaw = prev_yaw * SMOOTHING + (1 - SMOOTHING) * float(yaw)
            prev_scale = prev_scale * SMOOTHING + (1 - SMOOTHING) * float(scale)

        smoothed_nose = (int(prev_nose[0]), int(prev_nose[1]))
        smoothed_yaw = prev_yaw
        smoothed_scale = prev_scale
        center = np.array(smoothed_nose, dtype=np.float32)
        dst_points_2d_scaled = (dst_points_2d - center) * SWAP_SCALE + center

        warped = None

        if USE_EXPRESSION_WARP:
        
            warped_buffer.fill(0)
            for i1, i2, i3 in TRI_INDICES:
                t_src = [src_points_2d[i1], src_points_2d[i2], src_points_2d[i3]]
                t_dst = [dst_points_2d_scaled[i1], dst_points_2d_scaled[i2], dst_points_2d_scaled[i3]]
                warp_triangle(src_img, warped_buffer, t_src, t_dst)
            warped = np.clip(warped_buffer, 0, 255).astype(np.uint8)
        else:
           
            if USE_TPS and hasattr(cv2, "createThinPlateSplineShapeTransformer"):
                try:
                    tps = cv2.createThinPlateSplineShapeTransformer()
                    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points_2d))]
                    tps.estimateTransformation(
                        src_points_2d.reshape(-1, 1, 2),
                        dst_points_2d_scaled.reshape(-1, 1, 2),
                        matches
                    )
                    warped = tps.warpImage(src_img)
                except cv2.error:
                    warped = None

            if warped is None:
                M, _ = cv2.estimateAffinePartial2D(
                    src_points_2d[AFFINE_INDEXES],
                    dst_points_2d_scaled[AFFINE_INDEXES],
                    method=cv2.LMEDS
                )

                if M is None:
                    M = cv2.getRotationMatrix2D(
                        center=smoothed_nose,
                        angle=smoothed_yaw,
                        scale=smoothed_scale
                    )

                warped = cv2.warpAffine(src_img, M, (640, 480), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

  
        mask_buffer.fill(0)
        hull = cv2.convexHull(dst_points_2d_scaled.astype(np.int32))
        cv2.fillConvexPoly(mask_buffer, hull, 255, lineType=cv2.LINE_8)

        feature_mask_buffer.fill(0)
        left_eye_idx = LEFT_EYE_IDX + LEFT_IRIS_IDX
        right_eye_idx = RIGHT_EYE_IDX + RIGHT_IRIS_IDX

        left_eye = cv2.convexHull(dst_points_2d_scaled[left_eye_idx].astype(np.int32))
        right_eye = cv2.convexHull(dst_points_2d_scaled[right_eye_idx].astype(np.int32))
        cv2.fillConvexPoly(feature_mask_buffer, left_eye, 255, lineType=cv2.LINE_8)
        cv2.fillConvexPoly(feature_mask_buffer, right_eye, 255, lineType=cv2.LINE_8)

        if ALWAYS_USE_LIVE_MOUTH:
            lips_idx = LIPS_OUTER_IDX + LIPS_INNER_IDX
            lips = cv2.convexHull(dst_points_2d_scaled[lips_idx].astype(np.int32))
            cv2.fillConvexPoly(feature_mask_buffer, lips, 255, lineType=cv2.LINE_8)

        if FEATURE_DILATE > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FEATURE_DILATE, FEATURE_DILATE))
            cv2.dilate(feature_mask_buffer, k, dst=feature_mask_buffer, iterations=1)

        mask = cv2.subtract(mask_buffer, cv2.bitwise_and(mask_buffer, feature_mask_buffer))

        if MASK_BLUR > 1:
            cv2.blur(mask, (MASK_BLUR, MASK_BLUR), dst=mask)

        warped_blend = warped
        if USE_COLOR_MATCH:
            warped_blend = match_color_lab(warped_blend, frame, mask)

        if USE_SEAMLESS:
            x, y, w, h = cv2.boundingRect(hull)
            center = (x + w // 2, y + h // 2)
            try:
                frame = cv2.seamlessClone(warped_blend, frame, mask, center, cv2.NORMAL_CLONE)
            except cv2.error:
                mask_3 = (mask / 255.0).astype(np.float32)
                mask_3 = np.stack([mask_3, mask_3, mask_3], axis=2)
                frame = (warped_blend.astype(np.float32) * mask_3 + frame.astype(np.float32) * (1 - mask_3)).astype(np.uint8)
        else:
            mask_3 = (mask / 255.0).astype(np.float32)
            mask_3 = np.stack([mask_3, mask_3, mask_3], axis=2)
            frame = (warped_blend.astype(np.float32) * mask_3 + frame.astype(np.float32) * (1 - mask_3)).astype(np.uint8)

    if frame_count % DETECT_EVERY_N == 0:
        det_face = detect_extract_face(frame)
        if det_face is not None:
            detect_frame_buffer.append(detect_preprocess(det_face))

            if len(detect_frame_buffer) == DETECT_SEQ_LENGTH:
                seq = np.array(list(detect_frame_buffer))
                tensor = torch.FloatTensor(seq).unsqueeze(0).to(DETECT_DEVICE)

                with torch.no_grad():
                    output = detect_model(tensor)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    detect_confidence = probs[0][pred].item()

                if pred == 0:
                    detect_label = "REAL"
                    detect_color = (0, 255, 0)
                else:
                    detect_label = "FAKE"
                    detect_color = (0, 0, 255)
                
                print(f"Frame {frame_count}: {detect_label} (Confidence: {detect_confidence:.4f})")
            elif len(detect_frame_buffer) == 1:
            
                print(f"Buffering frames for detection... ({len(detect_frame_buffer)}/{DETECT_SEQ_LENGTH})")
            elif len(detect_frame_buffer) % 3 == 0:
                print(f"Buffering... ({len(detect_frame_buffer)}/{DETECT_SEQ_LENGTH})")
        elif frame_count % DETECT_EVERY_N == 0 and frame_count > 0:
            if frame_count <= DETECT_EVERY_N * 5:  
                print(f"Frame {frame_count}: No face detected for analysis")

    cv2.putText(frame, f"{detect_label} ({detect_confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, detect_color, 2)

    cv2.imshow("3D-Aware Live Face Swap + Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

print("\n" + "="*60)
print("Face Swap Detection Ended")
print(f"Total frames processed: {frame_count}")
print("="*60 + "\n")

if cap is not None:
    cap.release()
cv2.destroyAllWindows()