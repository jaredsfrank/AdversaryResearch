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


def test(inputs):
  yield inputs


class Adversary(object):

  def __init__(self):
    self.means = [0.485, 0.456, 0.406]
    self.stds = [0.229, 0.224, 0.225]
    self.path = "/scratch/datasets/imagenet/val"

  def imshow(self, img):
    """Normalizes and displays img."""
    # TODO: Get rid of loop. Use indexing
    for i in range(len(means)):
      img[i, :, :] *= self.stds[i]
      img[i, :, :] += self.means[i]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

  def load_data(self, path, batch_size, shuffle = True):
    """Returns data loader object given a path to images and a batch size."""
    normalize = transforms.Normalize(self.means,
             self.stds)
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

  def is_done(self, predicted, target_class, batch_size, iters, min_iters):
    all_right = np.all(predicted.numpy() == [target_class]*batch_size)
    return all_right and iters > min_iters 

  def create_adversary(batch_size, target_class, image_reg, lr, l_inf=False):
    # Load pretrained network
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    # Load in first <batch_size> images for validation
    val_loader = load_data(self.path, batch_size, True)
    data = next(iter(val_loader))
    images, _ =  data
    old_image = images.clone()
    inputs = Variable(images, requires_grad = True)
    new_labels = Variable(torch.LongTensor([target_class]*batch_size))
    # Instantiate Loss Classes
    CrossEntropy = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    opt = optim.SGD(test(inputs), lr=lr, momentum=0.9)
    save_figure(images, "Before_{}_{}".format(image_reg, lr))
    predicted = torch.Tensor([-1]*batch_size)
    iters = 0
    min_iters = 1000
    while not self.is_done(predicted, target_class, batch_size, iters, min_iters):
      print "Iteration {}".format(iters)
      outputs = resnet(inputs)
      model_loss = CrossEntropy(outputs, new_labels)
      image_loss = MSE(inputs, Variable(old_image))
      # image_loss = torch.max(inputs - Variable(old_image))
      loss = model_loss + image_reg*image_loss
      predicted = torch.max(outputs.data, 1)
      print "Target vs Predicted Class Weights:"
      print np.dstack((outputs.data[:, target_class].numpy(),
                      predicted[0].numpy()))
      predicted = predicted[1]
      iters += 1
      if not not self.is_done(predicted, target_class, batch_size, iters, min_iters):
        opt.zero_grad()
        loss.backward()
        opt.step()

    save_figure(inputs.data, "After_{}_{}".format(image_reg, lr))
    plt.show()











def save_figure(imgs, name):
  fig = plt.figure(name)
  imshow(torchvision.utils.make_grid(imgs))
  fig.savefig("/scratch/jsf239/{}.png".format(name))




def load_and_run_pretrained():
  # Loading pretrained network and data 
  resnet = models.resnet101(pretrained=True)
  resnet.eval()
  valdir = "/scratch/datasets/imagenet/val"
  val_loader = load_data(valdir, 2)
  print("Done instantiating data loader")
  correct = 0
  total = 0
  # Cycle through all data and determine which were correctly labeled
  for data in val_loader:
    images, labels = data
    output = resnet(Variable(images))
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
  print("Accuracy is {}".format(float(correct)/total))
  return float(correct)/total

