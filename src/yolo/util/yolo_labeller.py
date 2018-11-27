import pandas as pd
import os
import random
import cv2

# class mapping
mapping = {
    'articulated_truck': 0,
    'bicycle': 1,
    'bus': 2,
    'car': 3,
    'motorcycle': 4,
    'motorized_vehicle': 5,
    'non-motorized_vehicle': 6,
    'pedestrian': 7,
    'pickup_truck': 8,
    'single_unit_truck': 9,
    'work_van': 10 
}

# variables
GT_DATA_FILE = 'data/localization/MIO-TCD-Localization/gt_train.csv'
DATA_PATH = 'data/localization/MIO-TCD-Localization/train/'
#DATA_PATH = 'test_output/'

# read ground truth file
gt_data = pd.read_csv(GT_DATA_FILE, header=None, dtype={0: str})
gt_data.columns = ['image', 'label', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']

# create txt file for each image with following lines to defien detections [category number] [gt_x1] [gt_y1] [gt_x2] [gt_y2]
current_image = '00000000'
img = cv2.imread('data/localization/MIO-TCD-Localization/train/00000000.jpg')
height, width, _ = img.shape
current_labels = []
for i, row in gt_data.iterrows():  
    print('Processing object: {}/{}'.format(i, gt_data.shape[0]))
    if row['image'] != current_image:
        # write to txt file
        result = pd.DataFrame(current_labels)
        result.to_csv( '{}{}.txt'.format(DATA_PATH, current_image), index=False, header=False)
        # reset values
        current_image = row['image']
        img = cv2.imread('data/localization/MIO-TCD-Localization/train/{}.jpg'.format(current_image))
        height, width, _ = img.shape
        current_labels = []
    x1 = float(row['gt_x1'])
    y1 = float(row['gt_y1'])
    x2 = float(x1 + 1) if x1 == row['gt_x2'] else float(row['gt_x2'])
    y2 = float(y1 + 1) if y1 == row['gt_y2'] else float(row['gt_y2'])
    x_center = ((x1 + x2)/2)/width
    y_center = ((y1 + y2)/2)/height
    object_width = (x2 - x1)/width
    object_height =(y2 - y1)/height
    current_labels.append( [ '{} {} {} {} {}'.format(mapping[row['label']], x_center, y_center, object_width, object_height) ] )
