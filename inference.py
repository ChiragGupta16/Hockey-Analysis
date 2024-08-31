from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict("data/match.mp4",save= True)


