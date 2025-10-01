import torch
from torch.utils.data import TensorDataset, DataLoader


def ocen_model(model, X_test, y_test, categories, batch_size=32):
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 3, H, W)
    y_test = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    print(f"[INFO] Rozpoznawanie ras psów – liczba klas: {len(categories)}")
    model.eval()

    poprawne = 0
    wszystkie = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            poprawne += (predicted == labels).sum().item()
            wszystkie += labels.size(0)

    acc = poprawne / wszystkie
    print(f"[WYNIK] Dokładność na zbiorze testowym: {acc * 100:.2f}%")
    return acc

def pokaz_przyklady(model, X_test, y_test, categories, liczba=5):
    import matplotlib.pyplot as plt
    import random

    model.eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for i in range(liczba):
        idx = random.randint(0, len(X_test) - 1)
        img = torch.tensor(X_test[idx], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        label = y_test[idx]

        with torch.no_grad():
            output = model(img)
            predicted = torch.argmax(output).item()

        plt.imshow(X_test[idx])
        plt.title(f"Rzeczywista rasa: {categories[label]}\nRozpoznana jako: {categories[predicted]}")
        plt.axis('off')
        plt.show()