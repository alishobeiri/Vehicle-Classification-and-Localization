from yolo import Yolo
import pandas as pd
import cv2
import os

class Localizer:

    def __init__(self, verbose=True):
        self.yolo = Yolo('yolov3.cfg', 'yolov3.weights', 'yolov3.txt')
        self.iterator = 0

    def analyze_image(self, directory, ground_truths, name):
        image, indices, boxes = self.yolo.extract_objects('{}/{}.jpg'.format(directory, name))

        # crop, normalize, and save images while analyzing
        cropped = self.yolo.crop_objects(image, indices, boxes, 200, 200)
        for img in cropped:
            self.yolo.save_image(img, 'output/cropped', str(self.iterator))
            self.iterator +=  1

        return []

    def run(self, directory, ground_truth, output):
        """
        Runs the localizer across the dataset and outputs 
        a csv file that summarizes the results

        Args:
            directory: Path to images directory
            ground_truth: Path to ground truth file
            output: Output file name
        """
        gt_data = pd.read_csv(ground_truth, header=None, dtype={0: str})
        gt_data.columns = ['image', 'label', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']
        result = []

        currrent_detections = []
        current_image = gt_data['image'][0]
        for i, row in gt_data.iterrows():
            if i > 5:
                break
            if (current_image == row[0]):
                currrent_detections.append(row)
            else:
                output = self.analyze_image(directory, currrent_detections, current_image)
                current_image = row[0]

if __name__ == "__main__":
    localizer = Localizer()
    localizer.run('data/localization/MIO-TCD-Localization/train', 'data/localization/MIO-TCD-Localization/gt_train.csv', 'output/localizations.csv')