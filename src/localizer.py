from yolo.yolo import Yolo
import pandas as pd
pd.options.mode.chained_assignment = None
from svm.svm_predict import SVM
import cv2
import os
import numpy as np

class Localizer:

    def __init__(self, ground_truth, test_set, verbose=True):
        """
        Initializes localizer class used for running localization + classification through a dataset
        """
        # keep track of the current scoring metrics
        metrics = {
            'tp': 0, # number of true positives
            'fp': 0, # number of false positives
            'fn': 0, # number of false negatives
            'yolo': 0, # number of correct classifications by yolo
            'svm': 0, # number of correct classifications by svm
            'total': 0, # total number of images
            'non-detect': 0, # total number of images where no objects were detected
            'predicted-labels-yolo': [], # predicted yolo labels for true positives
            'predicted-labels-svm': [], # predicted svm labels for true positives
            'true-labels': [] # true labels for true positives
        }

        # load the test sets
        test_images = pd.read_csv(test_set)
        
        # load the ground truths
        gt_data = pd.read_csv(ground_truth, header=None, dtype={0: str})
        gt_data.columns = ['image', 'label', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']

        self.metrics = metrics
        self.dim = 416
        self.test_images = test_images
        self.gt_data = gt_data
        self.yolo = Yolo('yolo/yolo-custom.cfg', 'yolo/yolo-custom22288.weights', 'yolo/yolo-custom.txt')
        self.svm = SVM('svm/Pretrained/svm_k_4.joblib')
        self.iterator = 0

    def analyze_image(self, image_directory, name):
        """
        Uses yolo model to predict the bounding boxes and labels of an image.

        Args:
            image_directory: Path to the image directory
            name: Name of the image

        Returns:
            detections: A list of detections in the same format as the gt_train.csv file
            image: The image which the analyzation occured on
        """
        image, indices, boxes, class_ids = self.yolo.extract_objects('{}/{}'.format(image_directory, name))
        detections = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            label = str(self.yolo.classes[class_ids[i]]) 
            detections.append([name.split('.')[0], label, round(x), round(y), round(x+w), round(y+h)])
        return detections, image

    def draw_bounding_boxes(self, detections, ground_truth, image):
        """
        Draws the bounding boxes of the objects on an image for demo purposes

        Args:
            detections: List of detections given in same format as gt_train.csv file
            ground_truth: List of the ground truths to use for comparison
            image: Image object of the detection

        Return:
            result_pred: Image with predictions drawn
            result_gt: Image with ground truth boxes drawn
        """
        result_pred = image.copy()
        result_gt = image.copy()
        for d in detections:
            cv2.rectangle(result_pred, (d[2],d[3]), (d[4],d[5]), (0,255,0), 2)
            cv2.putText(result_pred, d[1], (d[2]-10,d[3]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        for i, row in ground_truth.iterrows():               
            cv2.rectangle(result_gt, (row['gt_x1'],row['gt_y1']), (row['gt_x2'],row['gt_y2']), (0,255,0), 2)
            cv2.putText(result_gt, row['label'], (row['gt_x1']-10,row['gt_y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        return result_pred, result_gt

    def calculate_overlap(self, pred, gt):
        """
        Calculates the pixel overal between two detections in an image

        Args:
            pred: Coordinates for the prediction [x1, y1, x2, y2]
            gt: Coordinates for the ground truth [x1, y1, x2, y2]

        Returns:
            percent_overlap: Overlap in percent value
        """
        # create binary images with object pixels set to 1
        image_pred = np.full((self.dim, self.dim), False)
        image_pred[pred[1]:pred[3],pred[0]:pred[2]] = True

        image_gt = np.full((self.dim, self.dim), False)
        image_gt[gt[1]:gt[3],gt[0]:gt[2]] = True

        # perform AND operation to calculate overlap
        result = np.bitwise_and(image_gt, image_pred)
        percent_overlap = np.sum(result) / np.sum(image_gt)

        # return the sum of array (number of 1s)
        return percent_overlap

    def classify_svm(self, detection, image):
        cropped = image[detection[3]:detection[5], detection[2]:detection[4]]
        pred = self.svm.predict(cropped)
        return str(pred[0]).split("'")[1]
        
    
    def calculate_score(self, detections, ground_truth, image):
        """
        Calculates the localization score of the prediction and updates metrics values

        Args:
            detections: List of detections given in same format as gt_train.csv file
            ground_truth: List of the ground truths to use for comparison
            image: The original image which that detection was run on
        """
        tp, fp, fn = 0, 0, 0
        detections_copy = detections.copy()
        match_threshold = 0.7
        correct_classifications_yolo = 0
        correct_classifications_svm = 0
        for i, row in ground_truth.iterrows():
            gt = [ row['gt_x1'], row['gt_y1'], row['gt_x2'], row['gt_y2'] ]
            # find maximum overlap
            max_overlap = [0.0, []]
            for d in detections_copy:
                pred = d[2:6]
                overlap = self.calculate_overlap(pred, gt)
                if overlap > max_overlap[0]:
                    max_overlap = [overlap, d]
            # if max overlap exceed threshold, then mark as true positive
            if max_overlap[0] > match_threshold:
                tp += 1
                self.metrics['predicted-labels-yolo'].append(d[1])
                self.metrics['true-labels'].append(row['label'])
                # score yolo classification
                if max_overlap[1][1] == row['label']:
                    self.metrics['yolo'] += 1
                    correct_classifications_yolo += 1
                # score svm classification
                svm_classification = self.classify_svm(max_overlap[1], image)
                if svm_classification == row['label']:
                    self.metrics['svm'] += 1
                    correct_classifications_svm += 1
                detections_copy.remove(max_overlap[1])
            else:
                fn += 1
        # mark any other detections as false positive
        fp = len(detections) - tp

        # update global metrics
        self.metrics['tp'] += tp
        self.metrics['fp'] += fp
        self.metrics['fn'] += fn
        print('tp: {}, fp: {}, fn: {}'.format(tp,fp,fn))
        print('svm: {}/{}'.format(correct_classifications_svm, tp))
        print('yolo: {}/{} \n'.format(correct_classifications_yolo, tp))

        # print latest total metrics
        print('\nTotal Metrics - tp: {}, fp: {}, fn: {} '.format(self.metrics['tp'],self.metrics['fp'],self.metrics['fn']))
        print('Correct Classifications (YOLO): {}/{}'.format(self.metrics['yolo'], self.metrics['tp']))
        print('Correct Classifications (SVM): {}/{} \n'.format(self.metrics['svm'], self.metrics['tp']))

    def evaluate_prediction(self, detections, image, image_path, display=False):
        """
        Evaluates the result of a prediction by validating the prediction against the
        ground truth and computing DICE scores for localization.

        Args:
            detections: The list of detections computed by YOLO
            image: The image object which the detection was performed on
            image_path: The path to the actual image
            display: Whether to show the display for visualization (False by default)
        """
        # load ground truth for image and scale to 416x416
        ground_truth = self.gt_data.loc[self.gt_data['image'] == detections[0][0]]
        height, width, _ = (cv2.imread(image_path)).shape

        # scale the ground truth coordinates to 416x416
        ground_truth['gt_x1'] = ground_truth['gt_x1'].apply(lambda x: int(x * self.dim / width))
        ground_truth['gt_x2'] = ground_truth['gt_x2'].apply(lambda x: int(x * self.dim / width))
        ground_truth['gt_y1'] = ground_truth['gt_y1'].apply(lambda x: int(x * self.dim / height))
        ground_truth['gt_y2'] = ground_truth['gt_y2'].apply(lambda x: int(x * self.dim / height))

        self.calculate_score(detections, ground_truth, image)

        # show bounding boxes if display is set to true
        if display:
            image_pred, image_gt = self.draw_bounding_boxes(detections, ground_truth, image)
            cv2.imshow('Predictions: {}'.format(detections[0][0]), image_pred)
            cv2.imshow('Ground Truth: {}'.format(detections[0][0]), image_gt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run(self, image_directory, output_directory):
        """
        Runs the localizer across the dataset and outputs 
        all necessary results needed for the report.

        Args:
            image_directory: Path to images directory
            output_directory: Output file name
        """
        # store the final prediction results for yolo only and yolo+SVM
        predictions_yolo = []
        predictions_yolo_svm = []

        # iterate through each test image 
        for i, row in self.test_images.iterrows():
            print('Analyzing image {}/{}'.format(i, self.test_images.shape[0]))
            self.metrics['total'] += 1
            if i == 100:
                break
            # run yolo and retrieve results from image
            detections, image = self.analyze_image(image_directory, row[0].split('/')[-1])
            # evaluate the prediction result
            if len(detections) > 0:
                self.evaluate_prediction(detections, image, '{}/{}'.format(image_directory, row[0].split('/')[-1]), display=False)
                # add detections to result list
                for d in detections:
                    predictions_yolo.append(d)
            else:
                self.metrics['non-detect'] += 1
        
        # write predictions to a csv file
        predictions_yolo = pd.DataFrame(predictions_yolo)
        predictions_yolo_svm = pd.DataFrame(predictions_yolo_svm)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        predictions_yolo.to_csv( '{}/predictions_yolo.csv'.format(output_directory), index=False, header=False)
        predictions_yolo_svm.to_csv( '{}/predictions_yolo_svm.csv'.format(output_directory), index=False, header=False)

if __name__ == "__main__":
    localizer = Localizer('data/localization/MIO-TCD-Localization/gt_train.csv', 'yolo/test.txt')
    localizer.run('data/localization/MIO-TCD-Localization/train', 'output')