"""
File: predict.py
------------------------------------
Detect wheats using the best weights trained (best.pt). Retrieve a random image from the test dir for detection
which is saved as output.png

"""

#Import statements 
from ultralytics import YOLO #Import Yolo model
import os #Import os
from os import listdir #List all files in dir
import random #Randomised selection
from PIL import Image #Image Proc/Manipulation Libary

#Global Constants
DATA_PATH = "..\data\global-wheat-detection-data"

def retrieve_path():
    """
    Returns the relative path of a selected item from test dir
    """
    images_list_stem = listdir(os.path.join(DATA_PATH,f"test"))
    selected_iamge_stem = random.choice(images_list_stem)

    selected_image_path = os.path.join(DATA_PATH,f"test\{selected_iamge_stem}")
    print(selected_image_path)
    return selected_image_path

def output_image(selected_image_path):
    """
    Perform inference on one test image and save the results (objecct detection)
    """
    img = Image.open(selected_image_path)

    model = YOLO("../runs/detect/train/weights/best.pt")

    #Returns a list of result
    res_bgr = model.predict(img)[0]
    #print(res.boxes)

    res_bgr = res_bgr.plot(line_width=1)
    #convert from bgr format to rgb
    res_rgb = res_bgr[:,:, ::-1]
    res_rgb = Image.fromarray(res_rgb)
    res_rgb.save("..\output.png")


if __name__ == "__main__":
    selected_image_path = retrieve_path()
    output_image(selected_image_path)


