import os
import kagglehub
import shutil

def pobierz_dane():
    path = os.path.join("dane", "pieski")

    if os.path.exists(path) and os.listdir(path):
        print("[INFO] Dane już istnieją w lokalnej ścieżce:", path)
        return path

    print("[INFO] Pobieranie danych z KaggleHub...")

    kagel = kagglehub.dataset_download("gpiosenka/70-dog-breedsimage-data-set")
    print("[INFO] Dane zostały pobrane do:", kagel)

    os.makedirs("dane", exist_ok=True)
    shutil.copytree(kagel, path, dirs_exist_ok=True)

    print("[INFO] Dane zostały skopiowane do:", path)
    return path