from svm_train_k_fold import hog, train_and_predict, get_batch
from sklearn.metrics import f1_score

def param_tuning(data_path, orientation_list, cell_size_list, block_size_list, img_path, output_dir):
    img_path = join(data_path, 'train')
    df = pd.read_csv(join(data_path, 'gt_train.csv'), names=["Id", "Category"])
    df = df.sample(frac=0.05)
    train_set = shuffle(df.iloc[fold[0]]).reset_index(drop=True)
    test_set = shuffle(df.iloc[fold[1]]).reset_index(drop=True)

    unique_classes = train_set["Category"].unique()
    num_classes = unique_classes.shape[0]

    batch_size = 10
    best_parameters = (0, 0, 0)
    best_f1_score = 0

    gt_label = test_set["Category"]

    for orientation in orientation_list:
        for block_size in block_size_list:
            for cell_size in cell_size_list:
                feature_param = {
                    'orientation': orientation,
                    'block_size': block_size,
                    'cell_size': cell_size
                }
                svm = SGDClassifier(penalty='l2')
                img_path = "{}/Plots/confusion_{}.jpg".format(output_dir, k)

                pred = train_and_predict(svm, 
                                         train_df, 
                                         test_df, 
                                         batch_size, 
                                         img_path,
                                         feature_param=feature_param)
                f1 = f1_score(pred, gt_label)
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_parameters = (orientation, block_size, cell_size)

if __name__ == "__main__":
    data_path = "../../../MIO-TCD/MIO-TCD-Classification/
    output_dir = "./feature_extraction/"
