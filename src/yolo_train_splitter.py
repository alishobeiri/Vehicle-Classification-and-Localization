import pandas as pd
import os
import random

# variables
DATA_PATH = 'data/localization/MIO-TCD-Localization/train/'

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