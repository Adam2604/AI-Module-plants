from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="dataset.yaml", epochs=25,  batch=8)
