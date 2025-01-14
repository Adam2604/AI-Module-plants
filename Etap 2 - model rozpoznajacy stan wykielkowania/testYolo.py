from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Ładowanie modelu
model = YOLO("runs/detect/train40/weights/best.pt")  # Załaduj właściwy model

# Lista ścieżek do testowych zdjęć
image_paths = [
    "C:/Model001/test/test1.jpg",
    "C:/Model001/test/test6.jpg",
    "C:/Model001/test/test8.jpg",
    "C:/Model001/test/test9.jpg",
    "C:/Model001/test/test10.jpg",
    "C:/Model001/test/test_3dzien1.jpg",
    "C:/Model001/test/test_3dzien2.jpg",
    "C:/Model001/test/test_3dzien2.jpg"
]

# Ustawienie progu zaufania
confidence_threshold = 0.1  # Zmniejsz lub zwiększ ten próg wedle potrzeb

# Iteracja po zdjęciach
for image_path in image_paths:
    print(f"Przeprowadzam detekcję na obrazie: {image_path}")
    
    # Wczytaj obraz
    img = cv2.imread(image_path)

    # Wykonaj detekcję
    results = model.predict(source=image_path, conf=0.05, iou=0.1, imgsz=640)

    # Zwykle wyniki są w pierwszym elemencie
    result = results[0]

    # Wyświetlanie wyników detekcji
    result.show()
    result.save()  # Zapisuje wynik w folderze `runs/detect/exp`

    # Wyświetlenie informacji o wykryciach
    for box in result.boxes:
        # Jeśli confidence > threshold, wyświetl detekcję
        if box.conf > confidence_threshold:
            # Pobieranie współrzędnych z tensora
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Konwersja do listy z wartościami

            # Przekształcanie wartości tensorów na liczby zmiennoprzecinkowe
            class_label = int(box.cls.item())  # Konwersja do liczby całkowitej
            confidence = box.conf.item()  # Konwersja do liczby zmiennoprzecinkowej

            # Prawdopodobieństwo dla klasy
            class_prob = box.conf.item()  # Wartość zaufania może działać jako prawdopodobieństwo

            # Tworzenie etykiety z klasą i jej prawdopodobieństwem
            label = f"Class {class_label}: {confidence:.2f}, Prob: {class_prob:.2f}"
            
            # Rysowanie prostokąta i tekstu
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            img = cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Wyświetlanie obrazu z wykryciami
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
