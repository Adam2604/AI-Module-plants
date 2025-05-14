from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Ładowanie modelu
model = YOLO("C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/runs/detect/train4/weights/best.pt")  # Załaduj właściwy model
#TRAIN40,42,50

#DO SAMYCH NASION 21, 27
# Lista nazw klas (zastąp odpowiednimi nazwami klas dla Twojego modelu)
class_names = ["Class_A", "Class_B"]

# Lista ścieżek do testowych zdjęć
image_paths = [
    #"C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/1.jpg",
    #'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/2.jpg',
    #
    'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test_9stycznia.jpg',
    'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/szalka1_9stycznia.jpg',
    # 'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/szalka8_9stycznia.jpg',
    # 'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/7_d.jpg',
    # 'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/4.jpg',
    # 'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/11_d.jpg',
    # 'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/14_w.jpg',
    # 'C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/47.jpg',
    # "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test2.jpg",
    # "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test6.jpg",
    # "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test8.jpg",
    # "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test9.jpg",
    # "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test10.jpg",
    # "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test_3dzien1.jpg",
    # "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test_3dzien2.jpg",
    # "C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/test/test_3dzien2.jpg"
]

# Ustawienie progu zaufania
confidence_threshold = 0.2  # Zmniejsz lub zwiększ ten próg wedle potrzeb

# Iteracja po zdjęciach
for image_path in image_paths:
    print(f"Przeprowadzam detekcję na obrazie: {image_path}")
    
    # Wczytaj obraz
    img = cv2.imread(image_path)

    # Wykonaj detekcję
    results = model.predict(source=image_path, conf=0.2, iou=0.1, imgsz=1024)

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

            # Zmieniamy nazwę klasy na odpowiednią nazwę z listy
            class_name = class_names[class_label]  # Pobierz nazwę klasy z listy
            label = f"{class_name}: {confidence:.2f}, Prob: {class_prob:.2f}"
            
            # Rysowanie prostokąta i tekstu
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            img = cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Wyświetlanie obrazu z wykryciami
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
