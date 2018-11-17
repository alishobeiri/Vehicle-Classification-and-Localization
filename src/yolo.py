import cv2
import argparse
import numpy as np

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
        conf_threshold = 0.5
        nms_threshold = 0.4

        # analyze result
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
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

        return image, indices, boxes

    def draw_bounding_boxes(self, image, indices, boxes):
        ''' 
        Draws the bounding boxes of the objects on an image for demo purposes

        Args:
            image (obj): Image object.
            indices (list): List of indices for each object
            boxes (list): List of coordinates for each object
        '''
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv2.rectangle(image, (round(x),round(y)), (round(x+w),round(y+h)), (0,255,0), 2)
    

def main():
    yolo = Yolo('yolov3.cfg', 'yolov3.weights', 'yolov3.txt')
    image, indices, boxes = yolo.extract_objects('dog.jpg')
    yolo.draw_bounding_boxes(image, indices, boxes)


    cv2.imshow("object detection", image)
    cv2.waitKey(0)
        
    cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



