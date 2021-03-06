from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from os.path import join
import cv2
import itertools
import time

from skimage.feature import hog as hog_sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
from joblib import dump, load

import warnings
warnings.filterwarnings('ignore')

def hog(img, orientations=8, block_size=5, cell_size=4, plot=False):
    """
    Performs HoG Feature Extraction. The parameters default 
    to best parmaeters, found through hyper parameter tunung. 
    Code can be found in /src/classifiers/feature_extraction.py
    
    Args: 
        img: Image to analyze. 
        orientations: Number of bins and orientations, defaults to 8.
        block_size: Block size, defaults to 5.
        cell_size: Cell size, defaults to 4. 
        plot: Boolean setting whether to show the extraction result, 
              defaults to False.
    """
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd = hog_sklearn(img_g, orientations=orientations,
                        pixels_per_cell = (cell_size, cell_size),
                        cells_per_block=(block_size, block_size),
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

def get_batch(df, index, batch_size, img_path, feature_param=None, viz=False):
    """
    Get current training or testing batch from dataset. 

    Args: 
        df: Dataset pandas dataframe.
        index: Starting index of the batch.
        batch_size: Batch size. 
        img_path: Path to the images directory. 
        feature_param: Feature extraction parameters, defaults to None. 
            Type: 
                dict() = { 
                            'orientations': orientations, 
                            'block_size': block_size,
                            'cell_size': cell_size 
                         }
        viz: Set whether to show HoG feature extraction map, defaults to False. 
    
    Returns: 
        batch features, batch labels. 
    """

    batch_df = df.iloc[index:index+batch_size]
    batch_img = []
    batch_labels = []
    for img_id, label in zip(batch_df["Id"], batch_df["Category"]):
        img_filename = img_path + "/" + label + "/" + str(img_id).zfill(8) + '.jpg'
        img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (64, 64))

            if feature_param:
                img = hog(img, 
                          orientations=feature_param['orientations'],
                          block_size=feature_param['block_size'],
                          cell_size=feature_param['cell_size'],
                          plot=viz)
            else:
                # Perform HoG with default tuned parameters. 
                img = hog(img, plot=viz)
            batch_img.append(img)
            batch_labels.append(label)
    batch_img = np.array(batch_img)
    batch_labels = np.array(batch_labels)
    return batch_img, batch_labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args: 
        cm: Confusion Matrix.
        classes: Unique classes label, 1D array like.
        normalize: boolean setting Normalization, defaults to False.
        title: Title of the confusion matrix, defaults to 'Confusion Matrix'. 
        cmap: Confusion matrix color scheme, defaults to Blue. 
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

def train_and_predict(clf, train_set, test_set, batch_size, img_path, feature_param=None, output=None):
    ''' 
    Trains the model and saves confusion matrix.

    Args:
        clf: Classifier
        train_set: Training set
        test_set: Test set 
        batch_size: Batch size 
        img_path: Path to image directory
        feature_param: Custom feature Extraction parameters, defaults to None.
        output: Output filename, defaults to None

    Returns: 
        Prediction labels, precision, accuracy and recall results
    '''
    s_time = time.time()
    unique_classes = train_set["Category"].unique()
    num_classes = unique_classes.shape[0]

    # Extract current training batch and fit classifier. 
    for i in range(0, train_set.shape[0], batch_size):
        features_train, train_label = get_batch(train_set, 
                                                i, 
                                                batch_size, 
                                                img_path, 
                                                feature_param=feature_param)

        features_train = features_train.reshape(features_train.shape[0], -1)
        clf.partial_fit(features_train, train_label, classes=unique_classes)


    # Get testing prediction results and ground truth labels. 
    all_test_label = []
    all_pred = []
    for i in range(0, test_set.shape[0], batch_size):
        features_test, test_label = get_batch(test_set, 
                                              i, 
                                              batch_size, 
                                              img_path, 
                                              feature_param=feature_param)

        features_test = features_test.reshape(features_test.shape[0], -1)
        
        preds = clf.predict(features_test)
        all_test_label.extend(test_label)
        all_pred.extend(preds)
    
    # Calculate precision, accuracy and recall metrics. 
    prec = precision_score(all_test_label, all_pred, average='micro')
    acc = accuracy_score(all_test_label, all_pred)
    recall = recall_score(all_test_label, all_pred, average='micro')
    print("prec: ", prec)
    print("acc: ", acc)
    print("recall: ", recall)
    cnf_matrix = confusion_matrix(all_pred, all_test_label)
    np.set_printoptions(precision=2)
    class_names = unique_classes

    # Plot normalized confusion matrix
    if output:
        plt.figure(figsize=(20, 20))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')
        plt.savefig(output)
        plt.show()
    f_time = time.time()
    print("Executed in: {}\n".format(f_time - s_time))
    
    return all_pred, prec, acc, recall

def k_fold_training(classifier, data_path, k, output_dir, save_model=True):
    ''' 
    Performs k_fold training on the dataset using SVM.

    Args:
        classifier: classification model (svm or logistic regression)
        data_path: path to the dataset
        k: number of k_fold cross-validation
        output_dir: output directory
        save_model: boolean define whether to 
                    save model for future use,
                    defaults to True
    '''
    img_path = join(data_path, 'train')
    df = pd.read_csv(join(data_path, 'gt_train.csv'), names=["Id", "Category"])
    df = df.head(30000)
    skf = StratifiedKFold(n_splits = k)
    k_fold = list(skf.split(df["Id"], df["Category"]))
    unique_classes = df["Category"].unique()
    num_classes = unique_classes.shape[0]

    preds = []
    batch_size = 10
    scores_tp_fp_fn = []
    prec_acc_recall = []

    # Create Plot and Measure folders if they don't exist already. 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(join(output_dir, 'Plots')):
        os.makedirs(join(output_dir, 'Plots'))
    if not os.path.exists(join(output_dir, 'Measures')):
        os.makedirs(join(output_dir, 'Measures'))
        
    for k, fold in enumerate(k_fold):
        s_time = time.time()
        print("Iteration/K-Valid: ", k)

        # Create SVM or Logistic Regression Classifiers 
        if classifier == 'svm':
            clf = SGDClassifier(penalty='l2')
        else: 
            clf = SGDClassifier(penalty='l2', loss='log')
        train_df = shuffle(df.iloc[fold[0]]).reset_index(drop=True)
        test_df = shuffle(df.iloc[fold[1]]).reset_index(drop=True)

        output = "{}/Plots/confusion_{}.jpg".format(output_dir, k)

        # Train classification and get validation results. 
        _, prec, acc, recall = train_and_predict(clf, 
                                                train_df, 
                                                test_df, 
                                                batch_size, 
                                                img_path, 
                                                output=output)
        
        if save_model:
            dump(svm, join(output_dir, classifier + '_k_' + str(k) + '.joblib'))
        
        prec_acc_recall.append((prec, acc, recall))
    np.savetxt(join(output_dir, 'Measures/prec_acc_recall.csv'), 
               prec_acc_recall, 
               delimiter=',', 
               header="precision,accuracy,recall")

def _parser():
    usage = ''
    parser = ArgumentParser(prog='classifier', usage=usage)
    parser.add_argument('-c', '--classifier', help='Classifier (svm or logreg)', required=True)
    parser.add_argument('-d', '--data', help='Data Path', required=True)
    parser.add_argument('-o', '--output', help='Output directory', required=True)
    parser.add_argument('-k', '--kfold', help='K fold', required=False)

    return parser


if __name__ == "__main__" :
    args = _parser().parse_args() 
    clf = args.classifier

    # Make sure that classifier is either SVM or Log Regression
    if not (clf == 'svm' or clf == 'logreg'):
        raise Exception('The classifier argument should either be "svm" or "logreg"')
    data_path = "../../../../data/Classification/"
    output_dir = "./test/"
    k = int(args.kfold) if args.kfold else 10

    k_fold_training(clf, data_path=args.data, k=k, output_dir=args.output, save_model=True)
