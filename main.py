import os

from src.loader import zaladuuj
from src.trainer import train
from src.evaluator import ocen_model, pokaz_przyklady
from src.predictator import predict

def main():
    # 1. Wczytaj dane z podzielonych folderów
    sciezka = "dane/pieski"
    X_train, y_train, categories = zaladuuj(os.path.join(sciezka, "train"))
    X_valid, y_valid, _ = zaladuuj(os.path.join(sciezka, "valid"), categories)
    X_test, y_test, _ = zaladuuj(os.path.join(sciezka, "test"), categories)
    #
    # # 2. trening + valid
    model = train(X_train, y_train, X_valid, y_valid, categories)
    #
    # # 3. ocena modelu na danych testowych
    ocen_model(model, X_test, y_test, categories)
    #
    # # 4. przykładowe predykcje
    pokaz_przyklady(model, X_test, y_test, categories, liczba=10)

    # 5. własny plik z pieskiem
    from src.model import ResNetCustom
    import torch

    model = ResNetCustom(num_classes=len(categories))
    checkpoint = torch.load("modele/cnn_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    predict("uwu_pieski/bunia.jpg", categories)

if __name__ == "__main__":
    main()