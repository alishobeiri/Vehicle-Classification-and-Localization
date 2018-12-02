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

Earlier iterations of the weights can be downloaded at the URLs below.

* **5000 Iterations:** [https://415.blob.core.windows.net/data/yolo-custom-5000.weights](https://415.blob.core.windows.net/data/yolo-custom-5000.weights)
* **10000 Iterations:** [https://415.blob.core.windows.net/data/yolo-custom-10000.weights](https://415.blob.core.windows.net/data/yolo-custom-10000.weights)
* **15000 Iterations:** [https://415.blob.core.windows.net/data/yolo-custom-15000.weights](https://415.blob.core.windows.net/data/yolo-custom-15000.weights)
* **20000 Iterations:** [https://415.blob.core.windows.net/data/yolo-custom-20000.weights](https://415.blob.core.windows.net/data/yolo-custom-20000.weights)

