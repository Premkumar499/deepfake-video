import cv2
import os

INPUT_DATASET = "dataset"
OUTPUT_DATASET = "dataset_frames"

os.makedirs(os.path.join(OUTPUT_DATASET, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DATASET, "fake"), exist_ok=True)

def extract_frames(video_path, output_folder, video_name):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = f"{video_name}_frame_{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    print(f"{video_name}: {frame_count} frames extracted")


for label in ["real", "fake"]:
    
    input_folder = os.path.join(INPUT_DATASET, label)
    output_folder = os.path.join(OUTPUT_DATASET, label)
    
    for video_file in os.listdir(input_folder):
        
        if video_file.endswith(".mp4"):
            
            video_path = os.path.join(input_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            extract_frames(video_path, output_folder, video_name)

print("DONE")