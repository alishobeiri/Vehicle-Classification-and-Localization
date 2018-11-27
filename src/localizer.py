from yolo.yolo import Yolo
import pandas as pd
pd.options.mode.chained_assignment = None
import cv2
import os

class Localizer:

    def __init__(self, ground_truth, verbose=True):
        test_images = pd.read_csv('yolo/test.txt')
        
        gt_data = pd.read_csv(ground_truth, header=None, dtype={0: str})
        gt_data.columns = ['image', 'label', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']

        self.dim = 416
        self.test_images = test_images
        self.gt_data = gt_data
        self.yolo = Yolo('yolo/yolo-custom.cfg', 'yolo/yolo-custom11160.weights', 'yolo/yolo-custom.txt')
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
            image: Image object of the detection

        Return:
            image: Image with bounding boxes
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

    def evaluate_prediction(self, detections, image, image_path):
        # load ground truth for image and scale to 416x416
        ground_truth = self.gt_data.loc[self.gt_data['image'] == detections[0][0]]
        height, width, _ = (cv2.imread(image_path)).shape

        # scale the ground truth coordinates to 416x416
        ground_truth['gt_x1'] = ground_truth['gt_x1'].apply(lambda x: int(x * self.dim / width))
        ground_truth['gt_x2'] = ground_truth['gt_x2'].apply(lambda x: int(x * self.dim / width))
        ground_truth['gt_y1'] = ground_truth['gt_y1'].apply(lambda x: int(x * self.dim / height))
        ground_truth['gt_y2'] = ground_truth['gt_y2'].apply(lambda x: int(x * self.dim / height))

        # draw bounding boxes
        image_pred, image_gt = self.draw_bounding_boxes(detections, ground_truth, image)

        # display results
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
        for i, row in self.test_images.iterrows():
            print('Analyzing image {}/{}'.format(i, self.test_images.shape[0]))
            if i == 10:
                break
            detections, image = self.analyze_image(image_directory, row[0].split('/')[-1])
            self.evaluate_prediction(detections, image, '{}/{}'.format(image_directory, row[0].split('/')[-1]))
            for d in detections:
                predictions_yolo.append(d)
        
        # output prediction results
        predictions_yolo = pd.DataFrame(predictions_yolo)
        predictions_yolo_svm = pd.DataFrame(predictions_yolo_svm)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        predictions_yolo.to_csv( '{}/predictions_yolo.csv'.format(output_directory), index=False, header=False)
        predictions_yolo_svm.to_csv( '{}/predictions_yolo_svm.csv'.format(output_directory), index=False, header=False)

if __name__ == "__main__":
    localizer = Localizer('data/localization/MIO-TCD-Localization/gt_train.csv')
    localizer.run('data/localization/MIO-TCD-Localization/train', 'output')