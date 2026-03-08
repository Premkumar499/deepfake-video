import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset_loader import DeepfakeDataset
from model import DeepfakeModel
import torchvision.transforms as transforms
from tqdm import tqdm
import os



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"  

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4

NUM_WORKERS = 0 if os.name == 'nt' else 4


def train_epoch(model, train_loader, criterion, optimizer, scaler):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader)

    for frames, labels in loop:

        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        if USE_AMP:
            with torch.amp.autocast("cuda"):
                outputs = model(frames)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    accuracy = 100 * correct / total

    return total_loss, accuracy


def validate(model, val_loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for frames, labels in val_loader:

            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(frames)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return total_loss, accuracy



if __name__ == "__main__":

    print("Device:", DEVICE)


    dataset = DeepfakeDataset("dataset_faces")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))



    model = DeepfakeModel().to(DEVICE)
    for param in list(model.cnn.parameters())[-30:]:
        param.requires_grad = True


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        factor=0.5
    )

    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_val_acc = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)

        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%")

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(model.state_dict(), "best_deepfake_model.pth")

            print("Best model saved!")


    print("\nTraining Complete")