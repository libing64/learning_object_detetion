from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model
dataset_config = "/home/libing/dataset/Air_Plane_Detection.v3i.yolov8/data.yaml"
results = model.train(data=dataset_config, epochs=100, imgsz=640)
