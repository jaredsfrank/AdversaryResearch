import numpy as np
import pylab as plt


def convert_label(image_num):
	labels = np.loadtxt("imagenet_labels/imagenet_labels.csv", delimiter = ",", dtype = "string")
	order = np.loadtxt("imagenet_labels/labels_list.csv", delimiter = ",", dtype = "string")
	label = order[image_num-1]
	print(labels[np.where(labels[:,0] == label)[0]])



if __name__ == '__main__':
	convert_label(241)