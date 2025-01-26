import os
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

# image_dir = "/home/libing/dataset/tiny_coco_dataset/tiny_coco/train2017"
# image_dir = "/home/libing/dataset/AquariumDataset/AquariumCombined.v2i.voc/train"
image_dir = "/home/libing/dataset/Air_Plane_Detection.v3i.yolov8/train/images"

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

os.makedirs("result", exist_ok=True)

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    
    results = model([image_path])
    
    for result in results:
        boxes = result.boxes  
        masks = result.masks  
        keypoints = result.keypoints 
        probs = result.probs 
        obb = result.obb  
        
        result.show()
        
        result_filename = f"result/{image_file}"
        result.save(filename=result_filename)