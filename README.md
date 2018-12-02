# ECSE 415 - Introduction To Computer Vision Final Project

The following repository holds the source code for the ECSE 415 - Introduction To Computer Vision course final project. The final project involves completing both the classification and localization on the [MIO-TCD](http://podoce.dinf.usherbrooke.ca/challenge/dataset/) dataset with specific method restrictions. The exact requirements can be found in the attached PDF file.

## Prerequisites 

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

#### MIO-TCD Data

Both the localization and classification datasets will be used in the challenge. Download them directory from the website or through a linux terminal.

```
wget http://podoce.dinf.usherbrooke.ca/static/dataset/MIO-TCD-Classification.tar
wget http://podoce.dinf.usherbrooke.ca/static/dataset/MIO-TCD-Localization.tar
```

In addition to the provided datasets, an additional dataset was generated using the ground truth predictions for the localization dataset and can be downloaded [here](https://415.blob.core.windows.net/data/localizations_cropped.zip). The dataset can also be downloaded directly through a linex terminal.

```
wget https://415.blob.core.windows.net/data/localizations_cropped.zip
```

#### Python Packages

The necessary python packages can be installed by running the requirements.txt file using pip.

```
pip install -r requirements.txt
```

## Source Code 

All source code for the project can be found in the *src* folder. The main root folder contains three main scripts for outputting the necessary results to complete the goal of the challenge.

* **svm-classifier.py:** Runs the trained SVM model k-folds through the classification dataset.
* **log-reg-classifier.py:** Runs the trained logistic regression model k-folds through the classification dataset.
* **localizer:** Runs the trained yolo model through a test localization set while also feeding the localization outputs to the trained SVM classifier. This script outputs results for localization (using YOLO) in addition to classification (using both YOLO and SVM).

In addition to the main scripts are a set of directories used for various tasks:

* **classifier:** Contains all source code used for preprocessing, training, and testing the SVM and logistic regression classifiers.
* **yolo:** Contains all source code used for preprocessing, training, and testing the custom YOLO model.
* **results:** Output results used for the report, including screenshots of the performance of each model and a utility script used for additional plotting and organization of the results.



