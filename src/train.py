import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.model import HybridVisionTransformer


# -------------------------------
# Data loaders (Train + Validation)
# -------------------------------
def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


# -------------------------------
# Training function
# -------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)

    return avg_loss, accuracy


# -------------------------------
# Validation function
# -------------------------------
def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    data_dir = "Dog Emotion"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    model = HybridVisionTransformer(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader, val_loader = get_dataloaders(data_dir)

    epochs = 5

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_acc = validate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )
