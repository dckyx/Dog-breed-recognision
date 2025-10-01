import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.downloader import pobierz_dane

IMG_SIZE = 128

def zaladuuj(path, kategorie_zew=None):
    if not os.path.exists(path) or not os.listdir(path):
        print(f"[INFO] Folder {path} nie istnieje lub jest pusty. Próba pobrania danych...")
        pobierz_dane()

    x = []
    y = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono folderu: {path}")

    categories = kategorie_zew or sorted(os.listdir(path))

    print(f"[INFO] Wczytywanie danych z: {path}...")

    for category in categories:
        folder = os.path.join(path, category)
        if not os.path.isdir(folder):
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[UWAGA] Nie można wczytać: {img_path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            x.append(img)
            y.append(category)

    x = np.array(x) / 255.0
    le = LabelEncoder()
    le.fit(categories)
    y = le.transform(y)
    y = np.array(y)

    print(f"[INFO] Wczytano {len(x)} obrazów z {len(categories)} kategorii.")
    return x, y, categories