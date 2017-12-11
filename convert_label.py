import numpy as np
import pylab as plt


def convert_label(image_num):
	labels = np.loadtxt("imagenet_labels/imagenet_labels.csv", delimiter = ",", dtype = "str")
	order = np.loadtxt("imagenet_labels/labels_list.csv", delimiter = ",", dtype = "str")
	label = order[image_num-1]
	if image_num.shape:
		all_labels = []
		for l in label:
			all_labels.append((labels[np.where(labels[:,0] == l)[0]])[0,1])
		print(all_labels)
	else:
		print(labels[np.where(labels[:,0] == label)[0]])



if __name__ == '__main__':
	convert_label(np.array(241))
	convert_label(np.array([241,2,4]))