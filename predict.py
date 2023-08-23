from ultralytics import YOLO

path =  "C:\Users\Kelvin\Documents\GitHub\Computer_vision\runs\detect"
model_path = "\train\weights\last.pt"

# Loading the custom model
model =  YOLO(path+model_path)

# Making a prediction of the input
predict_path = ""
result = model(predict_path)

print(result)