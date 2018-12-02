from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
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
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
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

iteration = '10000'

# load class names
class_names = [
    'articulated_truck', 
    'bicycle', 
    'bus', 
    'car', 
    'motorcycle', 
    'motorized_vehicle', 
    'non-motorized_vehicle', 
    'pedestrian', 
    'pickup_truck', 
    'single_unit_truck', 
    'work_van'
]
# load predictions
predictions = np.loadtxt('predictions-{}.csv'.format(iteration), dtype='str', delimiter=',')

# sort into seperate arrays and compute accuracies for each class
svm_pred = []
yolo_pred = []
labels = []
for i in range(1, len(predictions)):
    svm_pred.append(predictions[i][0])
    yolo_pred.append(predictions[i][1])
    labels.append(predictions[i][2])

# compute precision and recalls
scores_svm = classification_report(labels, svm_pred, target_names=class_names)
scores_yolo = classification_report(labels, yolo_pred, target_names=class_names)
with open('scores-svm-{}.txt'.format(iteration), 'w') as f:
    f.write(scores_svm)
with open('scores-yolo-{}.txt'.format(iteration), 'w') as f:
    f.write(scores_yolo)

# compute confusion matrix
# svm_confusion = confusion_matrix(svm_pred, labels)
yolo_confusion = confusion_matrix(yolo_pred, labels)

# plot confusion matrix
# plt.figure(figsize=(8,8))
# plot_confusion_matrix(svm_confusion, classes=class_names, normalize=True,
#                       title='SVM Classification On Localization Dataset')
# plt.savefig('confusion-svm-{}.png'.format(iteration))

plt.figure(figsize=(8,8))
plot_confusion_matrix(yolo_confusion, classes=class_names, normalize=True,
                      title='YOLO Classification On Localization Dataset ({} Iterations)'.format(iteration))
plt.savefig('confusion-yolo-{}.png'.format(iteration))