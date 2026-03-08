import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

SEQ_LENGTH = 10
IMG_SIZE = 224

class DeepfakeDataset(Dataset):

    def __init__(self, root_dir):
        
        self.root_dir = Path(__file__).parent / root_dir
        self.samples = []
        for label in ["real", "fake"]:
            label_dir = self.root_dir / label
            if not label_dir.exists():
                raise FileNotFoundError(f"Directory not found: {label_dir}")
            files = sorted(os.listdir(label_dir))
            sequences = []
            for i in range(0, len(files) - SEQ_LENGTH):
                seq = files[i:i+SEQ_LENGTH]
                sequences.append(seq)
            for seq in sequences:
                self.samples.append(
                    (seq, 0 if label == "real" else 1, str(label_dir))
                )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        seq_files, label, label_dir = self.samples[idx]

        frames = []

        for file in seq_files:

            path = os.path.join(label_dir, file)

            img = cv2.imread(path)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            img = img / 255.0

            img = np.transpose(img, (2,0,1))

            frames.append(img)

        frames = np.array(frames)

        frames = torch.FloatTensor(frames)

        label = torch.LongTensor([label])

        return frames, label.squeeze()


if __name__ == "__main__":
    dataset = DeepfakeDataset("dataset_faces")
    print(len(dataset))          
    if len(dataset) == 0:
        print("No samples found. Run face_extraction.py first to populate dataset_faces/.")
    else:
        frames, label = dataset[0]   
        print(frames.shape)         
        print(label)                 