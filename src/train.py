"""
File: train.py
------------------------------------
Train the yolo model on custom wheat dataset (wheat.yaml points to dataset path)

"""

#Import statements
from ultralytics import YOLO #Import Yolo model
import shutil

def train_yolo(model):
    """
    Train the model. Image resize to 224 x 224 (shorter training time)
    """
    model.train(
        data="..\wheat.yaml",
        epochs=20,
        imgsz=(224,224), 
        batch=8
    ) #Could trial with Adam optimizer and tweak lr0 arguments as well

def move_output_files():
    """
    Move output files outside of src
    """
    shutil.move("yolov8n.pt", "..\yolov8n.pt")
    shutil.move("yolov10n.pt", "..\yolov10n.pt")
    shutil.move("runs",r"..\runs")

if __name__ == "__main__":
    #Initialise yolo model
    model = YOLO("yolov10n.pt")
    
    train_yolo(model)

    move_output_files()
