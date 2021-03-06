from sklearn import datasets, svm, metrics
from skimage.feature import hog as hog_sklearn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from joblib import dump, load

class SVM(object):
	def __init__(self, svm_filename):
		self.clf = load(svm_filename)

	def hog(self, img, plot=False):
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
	

	def predict(self, img):
		plt.imshow(img)
		img = cv2.resize(img, (64, 64))
		img_features = self.hog(img).reshape(1, -1)
		preds = self.clf.predict(img_features)
		preds = [i.decode('utf-8') for i in preds]
		if len(preds) == 1:
			return preds[0]
		return preds

