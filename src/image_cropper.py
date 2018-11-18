import pandas as pd
import cv2
import os

# variables
GT_DATA_FILE = 'data/localization/MIO-TCD-Localization/gt_train.csv'
DATA_PATH = 'data/localization/MIO-TCD-Localization/train/'
OUTPUT_DIR = 'output/'
WIDTH = 200
HEIGHT = 200

# read ground truth file
gt_data = pd.read_csv(GT_DATA_FILE, header=None, dtype={0: str})
gt_data.columns = ['image', 'label', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']

# stores the final mapping
mapping = []

# keeps track of the image number
n = 0

# at each iteration - crop image, normalize, and save with label
for i, row in gt_data.iterrows():
    print('Cropping object: {}/{}'.format(n, gt_data.shape[0]))
    image = cv2.imread('{}{}.jpg'.format(DATA_PATH, row['image']))
    cropped = image[row['gt_y1']:row['gt_y2'], row['gt_x1']:row['gt_x2']]
    normalized = cv2.resize(cropped, (WIDTH, HEIGHT))
    save_path = OUTPUT_DIR + 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + str(n) + '.jpg', normalized) 
    mapping.append([n, row['label']])
    n += 1

result = pd.DataFrame(mapping)
result.to_csv(OUTPUT_DIR + 'gt_cropped.csv', index=False, header=False)
    
