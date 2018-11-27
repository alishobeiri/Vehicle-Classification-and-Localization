''' Script for analyzing how many objects of each class there are '''
import pandas as pd 

# ground truth file location
GT_DATA_FILE = '../data/localization/MIO-TCD-Localization/gt_train.csv'

# object count map
object_count = {
    'articulated_truck': 0,
    'bicycle': 0,
    'bus': 0,
    'car': 0,
    'motorcycle': 0,
    'motorized_vehicle': 0,
    'non-motorized_vehicle': 0,
    'pedestrian': 0,
    'pickup_truck': 0,
    'single_unit_truck': 0,
    'work_van': 0 
}

# read data and update object count
gt_data = pd.read_csv(GT_DATA_FILE, header=None, dtype={0: str})
for i, row in gt_data.iterrows():  
    print('Processing row: {}/{}'.format(i, gt_data.shape[0]))
    object_count[row[1]] += 1

# save results
result = []
total = 0
for o in object_count:
    total += object_count[o]
    result.append([o, object_count[o]])
result.append(['Total', total])
result = pd.DataFrame(result)
result.to_csv('object_count.csv', index=False, header=False)



