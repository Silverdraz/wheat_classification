"""
File: munge_data.py
------------------------------------
Load from the raw dataset (global-wheat-detection-data) and process it to yolo format (normalsied coordinates)

"""

#Import statements
import os #
import ast #process string as code
import pandas as pd #dataframe
import numpy as np #arrays
import shutil #file operations
from sklearn.model_selection import train_test_split #train-val-test split
from tqdm import tqdm #progress bar

#Global constants
DATA_PATH = "..\data\global-wheat-detection-data"
OUTPUT_PATH = "..\data\wheat_data_annotated"

def train_val_df(df):
    """
    Split the df to create train, val datasets and return them accordingly
    """
    df_train, df_valid = train_test_split(
        df,
        test_size = 0.1,
        random_state=42,
        shuffle=True
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    return df_train, df_valid

def create_folders():
    """
    Create empty folders for files to be saved in them
    """
    os.makedirs("../data/wheat_data_annotated/images/train",exist_ok=True)
    os.makedirs("../data/wheat_data_annotated/images/validation",exist_ok=True)
    os.makedirs("../data/wheat_data_annotated/labels/train",exist_ok=True)
    os.makedirs("../data/wheat_data_annotated/labels/validation",exist_ok=True)

def process_data(data, data_type):
    """
    Process the data to normalised coordinates for yolo data format. 
    1. Save these annotations as labels
    2. Save the images from raw (train) to train and validation
    """
    for _,row in tqdm(data.iterrows(), total=len(data)):
        image_name = row['image_id']
        bounding_boxes = row["bboxes"]
        yolo_data = []
        for bbox in bounding_boxes:
            #[xmin, ymin, width, height] - bbox
            # width = 1024, height = 1024 - images
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            #Required format for yolo (normalised)
            x_center = x + w/2
            y_center = y + h/2
            x_center /= 1024.0
            y_center /= 1024.0
            w /= 1024.0
            h /= 1024.0
            yolo_data.append([0,x_center,y_center,w,h])
        yolo_data = np.array(yolo_data)
        np.savetxt(
            os.path.join(OUTPUT_PATH,f"labels\{data_type}\{image_name}.txt"),
            yolo_data,
            fmt=["%d","%f","%f","%f","%f"]
        )
        shutil.copyfile(
            os.path.join(DATA_PATH,f"train\{image_name}.jpg"),
            os.path.join(OUTPUT_PATH,f"images\{data_type}\{image_name}.jpg")
        )


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    print(df.head(), df.dtypes)

    #Convert from a str to a list of coordinates
    df.bbox = df.bbox.apply(ast.literal_eval)
    print(type(df.bbox[0]))
    
    #Group by image_id (every bbox is a row)
    df = df.groupby("image_id")["bbox"].apply(list).reset_index(name="bboxes")
    print("-" * 50)

    df_train, df_valid = train_val_df(df)
    
    create_folders()

    process_data(df_train, data_type = "train")
    process_data(df_valid, data_type = "validation")
    