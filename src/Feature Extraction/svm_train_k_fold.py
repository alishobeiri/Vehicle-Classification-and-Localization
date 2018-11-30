
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from skimage.feature import hog as hog_sklearn


# In[2]:


img_path = "../../../MIO-TCD/MIO-TCD-Classification/train"


# In[3]:


df = pd.read_csv("../../../MIO-TCD/MIO-TCD-Classification/gt_train.csv", names=["Id", "Category"])


# In[4]:


skf = StratifiedKFold(n_splits = 10)


# In[5]:


k_fold = list(skf.split(df["Id"], df["Category"]))


# For every dataset we need to do train test split, and from the train we need 20%  of data to train and validate neural network and remaining for training SVM

# In[6]:


unique_classes = df["Category"].unique()
num_classes = df["Category"].unique().shape[0]


# In[7]:


from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from joblib import dump, load
import time


# In[8]:


def hog(img, plot=False):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd = hog_sklearn(img_g, orientations=4,
                        pixels_per_cell = (3, 3),
                        cells_per_block=(3, 3),
                        block_norm='L1', 
                        multichannel=False,
                        feature_vector=True, 
                        visualize=plot)
    if plot:
        plt.figure(figsize = (10,10))
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title("Original Image"), plt.xticks([]), plt.yticks([])
        plt.show()
        plt.imshow(hog_image)
        plt.show()
    return fd

def get_batch(df, index, batch_size=10000, features=True):
    batch_df = df.iloc[index:index+batch_size]
    batch_img = []
    batch_labels = []
    for img_id, label in zip(batch_df["Id"], batch_df["Category"]):
        im_path = img_path + "/" + label + "/" + str(img_id).zfill(8) + '.jpg'
        img = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            if features:
                img = hog(img)
            batch_img.append(img)
            batch_labels.append(label)
    batch_img = np.array(batch_img)
    batch_labels = np.array(batch_labels)
    return batch_img, batch_labels

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        non_zero = cm > 0
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[9]:


import warnings
warnings.filterwarnings('ignore')

preds = []
batch_size = 10
scores_tp_fp_fn = []
prec_acc_recall = []
for j, fold in enumerate(k_fold):
    s_time = time.time()
    print("Iteration/K-Valid: ", j)
    svm = SGDClassifier(penalty='l2')
    train_df = shuffle(df.iloc[fold[0]]).reset_index(drop=True)
    test_df = shuffle(df.iloc[fold[1]]).reset_index(drop=True)
    
    for i in range(0, train_df.shape[0], batch_size):
        features_train, train_label = get_batch(train_df, i, batch_size)

        features = features_train.reshape(features_train.shape[0], -1)
        svm.partial_fit(features, train_label, classes=unique_classes)
    
    dump(svm, 'svm_k_' + str(j) + '.joblib')
    all_test_label = []
    all_pred = []
    for i in range(0, test_df.shape[0], batch_size):
        features_test, test_label = get_batch(test_df, i, batch_size)
        features = features_test.reshape(features_test.shape[0], -1)
        
        preds = svm.predict(features)
        all_test_label.extend(test_label)
        all_pred.extend(preds)
    
    prec = precision_score(all_test_label, all_pred, average='macro')
    acc = accuracy_score(all_test_label, all_pred)
    recall = recall_score(all_test_label, all_pred, average='macro')
    print("prec: ", prec)
    print("acc: ", acc)
    print("recall: ", recall)
    cnf_matrix = confusion_matrix(all_pred, all_test_label)
    np.set_printoptions(precision=2)
    class_names = unique_classes

    # Plot normalized confusion matrix
    plt.figure(figsize=(20, 20))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig("./Plots/confusion_"+str(j)+".jpg")
    plt.show()
    f_time = time.time()
    print("Executed in: ", f_time - s_time)
    print("\n")
    
    prec_acc_recall.append((prec, acc, recall))
np.savetxt("./Measures/prec_acc_recall.txt", prec_acc_recall)

