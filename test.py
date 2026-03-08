import cv2
from roboflow import Roboflow

rf = Roboflow(api_key="PdrEpLGdZ7y0FDDkdT7G")
project = rf.workspace("premkumar-3nya7").project("classification-e5kug")
model = project.version(1).model

video_path = "/home/premkumar/Downloads/psg/dataset/fake/fake.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1

    if frame_count % 30 == 0:
        
        frame_file = "frame.jpg"
        cv2.imwrite(frame_file, frame)
        
        result = model.predict(frame_file).json()
        
        prediction = result["predictions"][0]["top"]
        confidence = result["predictions"][0]["confidence"]
        
        print(f"Frame {frame_count}: {prediction} ({confidence:.2f})")

cap.release()