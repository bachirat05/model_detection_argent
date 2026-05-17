from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    name="money_detector",
    patience=10,
    device="cpu"
)

print("✅ Entraînement terminé !")
print("📁 Modèle : runs/detect/money_detector/weights/best.pt")