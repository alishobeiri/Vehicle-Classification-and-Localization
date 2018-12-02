import matplotlib.pyplot as plt
import pandas as pd 
from os.path import join

folders = ['svm', 'localization']
filename = 'prec_acc_recall.csv'
headers = ['precision', 'accuracy', 'recall']

metrics = {}

for folder in folders:
	# Store file in pandas frame.
	df = pd.read_csv(join(folder, filename), header=0)
	metrics[folder] = df

# Plot curve.
for header in headers: 
	plt.figure()
	plt.xlabel('k-fold')
	plt.ylabel(header)

	for folder in folders: 
		k_vals = metrics[folder]['k_val']
		k_vals = k_vals + 1
		y = metrics[folder][header]
		plt.plot(k_vals, y, label=folder)
	plt.ylim(0, 1)
	plt.legend(loc="lower right")
	plt.title('{} per k-folds'.format(header))
	plt.savefig('{}_comparison_curve.png'.format(header))
	plt.close()