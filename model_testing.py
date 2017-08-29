import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

def imshow(img):
    #img = img / 4 + 0.5     # unnormalize
    img[0, :, :] *= .229
    img[1, :, :] *= .224
    img[2, :, :] *= .225
    img[0, :, :] += .485
    img[1, :, :] += .456
    img[2, :, :] += .406
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def load_data(path, batch_size = 100, shuffle = True):
	normalize = transforms.Normalize([0.485, 0.456, 0.406],
					 [0.229, 0.224, 0.225])
	data_loader = torch.utils.data.DataLoader(
	        torchvision.datasets.ImageFolder(path, transforms.Compose([
	            transforms.Scale(256),
	            transforms.CenterCrop(224),
	            transforms.ToTensor(),
	            normalize,
	        ])),
	        batch_size=batch_size, shuffle=shuffle,
	        num_workers=1, pin_memory=True)
        return data_loader

def test(inputs):
    yield inputs


def create_adversary():
    resnet = models.resnet18(pretrained=True)
    valdir = "/scratch/datasets/imagenet/val"
    val_loader = load_data(valdir, 2, True)
    data = next(iter(val_loader))
    print("starting new batch")
    images, labels =  data
    print "The expected labels are {}".format(labels)
    old_image = images.clone()
    inputs = Variable(images, requires_grad = True)
    new_label = Variable(torch.LongTensor([1, 1]))
    crit = nn.CrossEntropyLoss()
    reg = nn.MSELoss()
    opt = optim.SGD(test(inputs), lr=.1, momentum=0.9)
    outputs = resnet(inputs)
    imshow(torchvision.utils.make_grid(images))
    plt.show()
    before_fig = plt.figure("before")
    imshow(torchvision.utils.make_grid(images))
    before_fig.savefig("/scratch/jsf239/before2.png")
    i = 0
    predicted = torch.Tensor([463])
    while not np.all(predicted.numpy() == [1, 1]):
        print "Iteration {}".format(i)
        i += 1
        outputs = resnet(inputs)
        l1 = crit(outputs, new_label)
        l2 = reg(inputs, Variable(old_image))

        loss = l1 + 1000*l2
        print crit(outputs, new_label), loss
        predicted = torch.max(outputs.data, 1)
        print outputs.data
        print predicted
        predicted = predicted[1]
        if np.all(predicted.numpy() == [1, 1]):
            after_fig = plt.figure("After")
            imshow(torchvision.utils.make_grid(inputs.data))
            after_fig.savefig("/scratch/jsf239/after2.png")
            diff_fig = plt.figure("Diff")
            imshow(torchvision.utils.make_grid(images-old_image))
            diff_fig.savefig("/scratch/jsf239/diff2.png")
            plt.show()
        else:
            opt.zero_grad()
            loss.backward()
            opt.step()
    

def load_and_run_pretrained():
	# Loading pretrained network and data 
	resnet = models.resnet101(pretrained=True)
	print resnet
	valdir = "/scratch/datasets/imagenet/val"
        val_loader = load_data(valdir, 2)
	print("Done instantiating data loader")

	correct = 0
	total = 0
	# Cycle through all data and determine which were correctly labeled
	for data in val_loader:
		images, labels = data
		# imshow(torchvision.utils.make_grid(images))
		plt.show()
		output = resnet(Variable(images))
                print ("The output is {}".format(output))
		_, predicted = torch.max(output.data, 1)
                print predicted
                print labels
		total += labels.size(0)
		correct += (predicted == labels).sum()
		print("Testing batch. Accuracy is {}".format(float(correct)/total))
	
	print("Accuracy is {}".format(float(correct)/total))

if __name__ == "__main__":
    # load_and_run_pretrained()
    create_adversary()
