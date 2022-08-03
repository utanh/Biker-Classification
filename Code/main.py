import torch

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
model_path = "best.pt"
model = torch.hub.load( "./yolov5", "custom", model_path, source="local")

# Set thresold 
model.conf = 0.7
# Images
img = 'test_img/img_UIT_120.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)
detection = results.xyxy[0].cpu().numpy()

# Results
print(detection.shape)

count = detection.shape[0]
print()
#results.show()
