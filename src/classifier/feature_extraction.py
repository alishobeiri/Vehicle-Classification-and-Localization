from svm_train_k_fold import hog, train_and_predict, get_batch
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
import csv
import os
from os.path import join
import pandas as pd

def param_tuning(data_path, orientation_list, cell_size_list, block_size_list, output_dir):
    img_path = join(data_path, 'train')
    df = pd.read_csv(join(data_path, 'gt_train.csv'), names=["Id", "Category"])
    df = df.sample(frac=0.05)
    split = int(len(df.index) * 0.8)
    train_set = shuffle(df.iloc[:split]).reset_index(drop=True)
    test_set = shuffle(df.iloc[split:]).reset_index(drop=True)

    print('total training samples: ', len(train_set))
    print('total test samples: ', len(test_set))

    unique_classes = train_set["Category"].unique()
    num_classes = unique_classes.shape[0]

    batch_size = 10
    best_parameters = (0, 0, 0)
    best_f1_score = 0

    gt_label = test_set["Category"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    writer = csv.writer(open(join(output_dir, 'feature_extraction_param.csv'), mode='w'))
    writer.writerow('orientation,block_size,cell_size,f1')
    for orientation in orientation_list:
        for block_size in block_size_list:
            for cell_size in cell_size_list:
                feature_param = {
                    'orientations': orientation,
                    'block_size': block_size,
                    'cell_size': cell_size
                }
                svm = SGDClassifier(penalty='l2')

                pred = train_and_predict(svm, 
                                         train_set, 
                                         test_set, 
                                         batch_size, 
                                         img_path,
                                         feature_param=feature_param)[0]
                f1 = f1_score(pred, gt_label, average='micro')
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_parameters = (orientation, block_size, cell_size)

                    print('new best f1 score', best_f1_score)
                    print('new best param', best_parameters)
                    print('\n')
                writer.writerow([orientation, block_size, cell_size, f1])

    print('\nfinal best parameters', best_parameters) 
    
    writer.writerow([])
    writer.writerow([best_parameters[0], best_parameters[1], best_parameters[2], best_f1_score])

if __name__ == "__main__":
    data_path = "../../../../data/Classification/"
    output_dir = "./feature_extraction/"
     
    orientation_list = list(range(4, 9))
    cell_size_list = [3, 4, 5]
    block_size_list = [3, 4, 5]
    param_tuning(data_path, orientation_list, cell_size_list, block_size_list, output_dir)
