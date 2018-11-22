from yolo import Yolo
import numpy as np
import pandas as pd
import cv2
import os

def compute_detections(directory, ground_truth_filename, output):
    ''' 
    Computes the total number of detections
    in the data set for each image in the dataset
    and compares it to its ground truth.

    Args:
        directory: Path to images directory
        output: Output file name
    '''
    detections = []
    total = 0

    yolo = Yolo('yolov3.cfg', 'yolov3.weights', 'yolov3.txt')

    count = 1
    files = os.listdir(directory)
    for filename in files:
        _, indices, _ = yolo.extract_objects(directory + '/' + filename)
        num_objects = len(indices)
        detections.append([filename.split('.')[0],num_objects])
        total += num_objects
        print('Number of objects detected in {}: {}'.format(filename, num_objects))

        if count % 10 == 0: 
            print("\nProcessed {} out of {}".format(count, len(files)))
        
        count += 1
    
    detections.append(['Total Objects Detected', total])

    predictions = pd.DataFrame(detections)
    ground_truth = parse_ground_truth(ground_truth_filename)
    result = pd.merge(predictions, ground_truth, how='outer', left_on=0, right_on=0)
    result['1_x'] = result['1_x'].fillna(0).astype(int)
    
    result.to_csv(output, index=False, header=False)

def parse_ground_truth(filename):
    ''' 
    Computes the total number of detections
    in the ground truth results for each image

    Args:
        filename: Path to ground truth csv file

    Returns: 
        result: Pandas dataframe with the ground truth number 
           object detected
    '''
    data = pd.read_csv(filename, header=None, dtype={0: str})
    result = data.groupby([0]).size().reset_index(name=1)
    result.loc[result.shape[0]] = ['Total Objects Detected', result[1].sum()]
    return result

if __name__ == "__main__":
    compute_detections('data/localization/MIO-TCD-Localization/train', 'data/localization/MIO-TCD-Localization/gt_train.csv', 'output/data_analyzing/train_detections.csv')