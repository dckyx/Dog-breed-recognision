import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # nowa warstwa wyjsciowa

    def forward(self, x):
        return self.model(x)

def train(X_train, y_train, X_valid, y_valid, categories, batch_size=64, epochs=10, learning_rate=0.001):
    num_classes = len(categories)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).permute(0, 3, 1, 2)
    y_valid = torch.tensor(y_valid, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size)

    model = ResNetModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print("[INFO] Trening modelu PyTorch...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Walidacja
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total * 100

        print(f"Epoka {epoch+1}/{epochs} - Strata: {avg_train_loss:.4f} - Val Accuracy: {accuracy:.2f}%")

    os.makedirs("modele", exist_ok=True)
    torch.save(model.state_dict(), "modele/cnn_model.pth")
    print("[INFO] Model zapisany jako: modele/cnn_model.pth")

    return model