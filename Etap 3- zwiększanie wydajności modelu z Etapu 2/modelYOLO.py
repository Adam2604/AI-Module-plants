from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="C:/Users/adamm/OneDrive/Dokumenty/Studia/Spektrum/Repozytorium/AI-Module-plants/Etap 3- zwiększanie wydajności modelu z Etapu 2/dataset.yaml",
        epochs=50,
        batch=14,
        #accumulate = 2,

        amp = True,
        cache = False,
        device=0,

        imgsz=1024,
        optimizer="Adam",
        lr0=0.0005,
        momentum=0.9,
        weight_decay=0.005,
        degrees=30,
        scale=0.7,
        shear=3.0,
        #hsv_h=0.03,
        #hsv_s=0.5,
        #hsv_v=0.3,
        flipud = 0.3,
        fliplr=0.5,
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
