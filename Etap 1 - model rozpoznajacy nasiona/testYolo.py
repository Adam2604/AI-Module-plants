from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Załaduj wytrenowany model
model = YOLO("C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/Modul-AI-roslinki/Etap 1 - model rozpoznajacy nasiona/runs/detect/train17/weights/best.pt")  # Zaktualizuj ścieżkę do swojego wytrenowanego modelu

# Lista ścieżek do testowych zdjęć
image_paths = [
    
    "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/Modul-AI-roslinki/Etap 1 - model rozpoznajacy nasiona/test/test4.jpg",
    "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/Modul-AI-roslinki/Etap 1 - model rozpoznajacy nasiona/test/test5.jpg",
    "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/Modul-AI-roslinki/Etap 1 - model rozpoznajacy nasiona/test/test6.jpg",
    # Dodaj kolejne ścieżki zdjęć
]

# Iteracja po zdjęciach
for image_path in image_paths:
    print(f"Przeprowadzam detekcję na obrazie: {image_path}")
    
    # Wczytaj obraz
    img = cv2.imread(image_path)

    # Przeprowadź detekcję
    results = model.predict(source=image_path, conf=0.3, iou=0.5, imgsz=1280)

    # Zwykle wyniki są w pierwszym elemencie
    result = results[0]

    # Zobacz wyniki
    result.show()  # Wyświetla obraz z zaznaczonymi wykryciami

    # Możesz także zapisać wynik
    result.save()  # Zapisuje wynik w folderze `runs/detect/exp`

    # Wyświetlenie informacji o wykryciach
    for box in result.boxes:
        print(f"Klasa: {box.cls}, Confidence: {box.conf}, Współrzędne: {box.xyxy}")

    # Wyświetlenie obrazu z wykrytymi nasionami
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
