from ultralytics import YOLO
import numpy

# load a pretrained model (medium model)
# To train a model from scratch, will have to use .yaml file format
model = YOLO("yolov8m.pt")

# Train the model
model.train(data="config.yaml", epochs=10)
metrics = model.val()


# predict on an image
# conf: Will be the minimum confidence of the model
# detection_output = model.predict(source= image_path , conf=0.4, save = True)

# # Display tensor array
# print(detection_output)

# # Print numpy array

# print(detection_output[0].numpy())
