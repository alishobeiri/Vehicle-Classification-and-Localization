# ECSE 415 - Introduction To Computer Vision Final Project

The following repository holds the source code for the ECSE 415 - Introduction To Computer Vision course final project. The final project involves completing both the classification and localization on the [MIO-TCD](http://podoce.dinf.usherbrooke.ca/challenge/dataset/) dataset with specific method restrictions. The exact requirements can be found in the attached PDF file.

## Prerequisites 

#### Python Packages

The necessary python packages can be installed by running the requirements.txt file using pip.

```
pip install -r requirements.txt
```

#### YOLO v3 (You Only Look Once)

A custom trained model of the YOLO v3 deep learning implementation is utilized for both localization and classification. The model was trained by using the provided ground truth data and following the instructions in the [Darknet github repository](https://github.com/AlexeyAB/darknet).

Download the [latest trained weights](https://415.blob.core.windows.net/data/yolo-custom-25000.weights) and place it in the *src/yolo* directory or download directly from a linux terminal.

```
wget https://415.blob.core.windows.net/data/yolo-custom-25000.weights
```


