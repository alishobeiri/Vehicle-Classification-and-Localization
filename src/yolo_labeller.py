import pandas as pd
import os
import random

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
current_labels = []
for i, row in gt_data.iterrows():  
    print('Processing object: {}/{}'.format(i, gt_data.shape[0]))
    if row['image'] != current_image:
        # write to txt file
        result = pd.DataFrame(current_labels)
        result.to_csv( '{}{}.txt'.format(DATA_PATH, row['image']), index=False, header=False)
        # reset values
        current_image = row['image']
        current_labels = []
    x1 = row['gt_x1']
    y1 = row['gt_y1']
    x2 = x1 + 1 if x1 == row['gt_x2'] else row['gt_x2']
    y2 = y1 + 1 if y1 == row['gt_y2'] else row['gt_y2']
    if x1 == 0:
        x1 += 1
        x2 += 1
    if y1 == 0:
        y1 += 1
        y2 += 1
    current_labels.append( [ '{} {} {} {} {}'.format(mapping[row['label']], float(x1), float(y1), float(x2), float(y2)) ] )
