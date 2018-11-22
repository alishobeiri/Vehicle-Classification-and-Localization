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
    current_labels.append( [ '{} {} {} {} {}'.format(mapping[row['label']], row['gt_x1'], row['gt_y1'], row['gt_x2'], row['gt_y2']) ] )

# create train/test test files
print('Creating training and testing sets...')
files = []
for filename in os.listdir(DATA_PATH):
    files.append(['{}{}'.format(DATA_PATH, filename)])
random.shuffle(files)
train = pd.DataFrame(files[:int(len(files)*0.75)])
test = pd.DataFrame(files[int(len(files)*0.75):])
train.to_csv('train.txt', index=False, header=False)
test.to_csv('test.txt', index=False, header=False)