import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F

IMG_SIZE = 128  # dopasuj do loadera

def predict(image_path, categories, model=None):
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    if model is None:
        from src.model import ResNetCustom
        checkpoint = torch.load("modele/cnn_model.pth", map_location=device)
        model = ResNetCustom(num_classes=len(categories))
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = categories[predicted.item()]

        import matplotlib.pyplot as plt

        plt.imshow(image)
        plt.title(f"Nazwa pliku: {image_path}\nRozpoznana jako: {predicted_class}")
        plt.axis('off')
        plt.show()

    print(f"[PREDYKCJA] Nazwa pliku: {image_path} – rozpoznana rasa psa: {predicted_class}")
    return predicted_class