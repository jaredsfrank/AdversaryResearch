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

class LBFGS(object):

  def __init__(self):
    self.mean_norm = [0.485, 0.456, 0.406]
    self.std_norm = [0.229, 0.224, 0.225]
    self.verbose = True

  def imshow(self, img):
      #img = img / 4 + 0.5     # unnormalize
      for i in range(len(self.mean_norm)):
        img[i, :, :] = img[i, :, :] * self.std_norm[i] + self.mean_norm[i]
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))

  def save_figure(self, imgs, name):
    fig = plt.figure(name)
    self.imshow(torchvision.utils.make_grid(imgs))
    fig.savefig("/scratch/jsf239/{}.png".format(name))

  def load_data(self, path, batch_size = 100, shuffle = True):
      normalize = transforms.Normalize(self.mean_norm,
                       self.std_norm)
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

  def diff(self, imgs, old_imgs):
    fig = plt.figure("diff")
    image_diff = np.abs(torchvision.utils.make_grid(imgs - old_imgs).numpy())
    plt.imshow(np.transpose(image_diff, (1, 2, 0)))

  def is_done(self, predicted, target_class, batch_size, iters, min_iters):
    all_right = np.all(predicted.numpy() == [target_class]*batch_size)
    return all_right and iters > min_iters

  def all_changed(self, original_labels, predictions):
    print "the diff is "
    print original_labels.numpy(), predictions.numpy()
    print original_labels.numpy() != predictions.numpy()
    print np.all(original_labels.numpy() != predictions.numpy())
    return np.all(original_labels.numpy() != predictions.numpy())

  def clamp_images(self, images):
    """Clamps image to between minimum and maximum range in place."""
    for i in range(len(self.mean_norm)):
      minimum_value = (0 - self.mean_norm[i])/self.std_norm[i]
      maximum_value = (1 - self.mean_norm[i])/self.std_norm[i]
      torch.clamp(images[:, i,:,:], min=minimum_value, max=maximum_value, out=images[:,i,:,:])

  def create_adversary(self, batch_size=1, target_class=1, image_reg=100, lr=.1):
      # Load pretrained network
      resnet = models.resnet18(pretrained=True)
      resnet.eval()
      for parameter in resnet.parameters():
          parameter.requires_grad = False
      # Load in first <batch_size> images for validation
      valdir = "/scratch/datasets/imagenet/val"
      val_loader = self.load_data(valdir, batch_size, True)
      data = next(iter(val_loader))
      images, labels =  data
      original_labels = labels.clone()
      inputs = Variable(images, requires_grad = True)
      new_labels = Variable(torch.LongTensor([target_class]*batch_size))
      # Instantiate Loss Classes
      CrossEntropy = nn.CrossEntropyLoss()
      MSE = nn.MSELoss()
      opt = optim.SGD(test(inputs), lr=lr, momentum=0.9)
      self.clamp_images(images)
      old_images = images.clone()
      if self.verbose:
        self.save_figure(old_images, "Before_{}_{}".format(image_reg, lr))
        plt.show()
        self.save_figure(images, "Before_{}_{}".format(image_reg, lr))
      predicted = labels
      iters = 0
      min_iters = 0
      while not self.all_changed(original_labels, predicted): # self.is_done(predicted, target_class, batch_size, iters, min_iters):
        if self.verbose:
          print "Iteration {}".format(iters)
        opt.zero_grad()
        self.clamp_images(images)
        outputs = resnet(inputs)
        model_loss = CrossEntropy(outputs, new_labels)
        image_loss = MSE(inputs, Variable(old_images))
        loss = model_loss + image_reg*image_loss
        predicted = torch.max(outputs.data, 1)
        if self.verbose:
          print "Target Class Weights Minus Predicted Weights:"
          print outputs.data[:, target_class] - predicted[0]
        predicted = predicted[1]
        iters += 1
        if not self.all_changed(original_labels, predicted): # self.is_done(predicted, target_class, batch_size, iters, min_iters):
            if self.verbose:
              self.save_figure(inputs.data, "After_{}_{}".format(image_reg, lr))
              self.diff(images, old_images)
              plt.show()
        else:
            loss.backward()
            opt.step()
        # image_loss = torch.max(inputs - Variable(old_images))
      return MSE(images, Variable(old_images))


