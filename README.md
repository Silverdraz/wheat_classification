# **Wheat Object Detection using Yolo V10**

## **About the Project**
The project aims to fine-tune Yolo V10 on a custom dataset (global wheat detection dataset), allowing object detection to be implemented even for objects or classes that are not included in the pre-training. This highlights the versatility of object detection on non pretrained images or classes.

## **Built With**
* Yolo V10 (Object Detection CV Framework)

## **Documentation/Overview of Architecture and folder structure**
The python script files are included in the **src section** for clarity, while the **data folders** contains both the raw images as well as the annotated files (yolo formmated labels). Firstly and specifically, the python script files are intended to be runned individually since it is broken into modules of **predict.py, train.py, munge_data.py.** These are technically modular files. Separately, the **runs folder** consist of the results, which shows that the loss decreases over epochs and map increases over epochs. (A positive sign). Most importantly, it consists of the best weights that is subsequently used for inference. **Wheat.yaml file** consits of the paths and config variables for training.

## Usage
The intended purpose is to showcase the finetuning capability of yolo v10.
