from ultralytics import YOLO
import numpy


# load a pretrained model 
model = YOLO("yolov8n.pt","v8")

image_path = r"C:\Users\Kelvin\Documents\GitHub\Coursera\OpenCV\Keene.jpg" #r"C:\Users\Kelvin\Documents\GitHub\Coursera\OpenCV\Streets.jpg"

# predict on an image
# conf: Will be the minimum confidence of the model
detection_output = model.predict(source= image_path , conf=0.4, save = True)

# Display tensor array
print(detection_output)

print(detection_output[0].numpy())