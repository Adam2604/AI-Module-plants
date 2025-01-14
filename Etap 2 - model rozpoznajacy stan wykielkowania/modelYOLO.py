from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(data="dataset.yaml", epochs=40,  batch=8, imgsz=1024, optimizer="Adam", lr0=0.0005)
