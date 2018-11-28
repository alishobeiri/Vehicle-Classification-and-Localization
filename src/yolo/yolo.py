import cv2
import argparse
import numpy as np
import os

class Yolo:
    
    def __init__(self, config_file, weights_file, classes_file, verbose=True):
        ''' 
        Initializes the YOLO class. Used for object localization.

        Args:
            config_file (str): Path to the yolo config file.
            weights_file (str): Path to yolo pre-trained weights.
            classes_file (str): Path to text file containing class names.
        '''
        classes = None
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        net = cv2.dnn.readNet(weights_file, config_file)
        
        self.classes = classes
        self.net = net

    def extract_objects(self, image_file):
        ''' 
        Extracts the object locations from an image.

        Args:
            image_file (str): Path to the image file.

        Return:
            objects (list): List of coordinates for each object detected
        '''
        # read image
        image = cv2.imread(image_file)
        image = cv2.resize(image, (416, 416)) 
        width = image.shape[1]
        height = image.shape[0]
        scale = 0.00392

        # feed throw network
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        output_layers = [self.net.getLayerNames()[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(output_layers)

        # set thresholds
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.2
        nms_threshold = 0.1

        # analyze result
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                #print('Class: {}. Confidence: {}'.format(class_id, confidence))
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        # remove any duplicates using non maximum supression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        return image, indices, boxes, class_ids

    def draw_bounding_boxes(self, image, indices, boxes, class_ids):
        ''' 
        Draws the bounding boxes of the objects on an image for demo purposes

        Args:
            image (obj): Image object.
            indices (list): List of indices for each object
            boxes (list): List of coordinates for each object
            class_ids (list): List of all the classes

        Return:
            image (obj): Image with bounding boxes
        '''
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            label = str(self.classes[class_ids[i]]) 
            cv2.rectangle(image, (round(x),round(y)), (round(x+w),round(y+h)), (0,255,0), 2)
            cv2.putText(image, label, (round(x)-10,round(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        return image

    def save_image(self, image, directory, filename):
        '''
        Saves the image to an output folder

        Args:
            image (obj): Image object.
            directory (str): Path of output directory.
            filename (str): Name of the file.
        '''
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = directory + '/' + filename + '.jpg'
        cv2.imwrite(path, image)

    def crop_objects(self, image, indices, boxes, height, width):
        '''
        Crops the objects out of the main image and normalizes to set dimensions

        Args:
            image (obj): Image object.
            indices (list): List of indices for each object
            boxes (list): List of coordinates for each object
            height (int): Normalized height.
            width(int): Normalize width.

        Return:
            images (list): List of cropped and normalized images.
        '''
        images = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cropped = image[round(y):round(y+h), round(x):round(x+w)]
            normalized = cv2.resize(cropped, (width, height)) 
            images.append(normalized)
        return images

def main():
    #yolo = Yolo('yolov3.cfg', 'yolov3.weights', 'yolov3.txt')
    yolo = Yolo('yolo-custom.cfg', 'yolo-custom.weights', 'yolo-custom.txt')
    image, indices, boxes, class_ids = yolo.extract_objects('../data/localization/MIO-TCD-Localization/train/00094811.jpg')

    # n = 0
    # cropped = yolo.crop_objects(image, indices, boxes, 200, 200)
    # for img in cropped:
    #     yolo.save_image(img, 'output', str(n))
    #     n = n + 1

    yolo.draw_bounding_boxes(image, indices, boxes, class_ids)
    cv2.imshow("object detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



