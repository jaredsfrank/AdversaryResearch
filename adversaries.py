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

  def __init__(self, batch_size):
    self.mean_norm = [0.485, 0.456, 0.406]
    self.std_norm = [0.229, 0.224, 0.225]
    self.verbose = False
    self.show_images = False
    valdir = "/scratch/datasets/imagenet/val"
    self.batch_size = batch_size
    self.val_loader = self.load_data(valdir, self.batch_size, True)
    # Instantiate Loss Classes
    self.CrossEntropy = nn.CrossEntropyLoss()
    self.MSE = nn.MSELoss()

  def imshow(self, img):
      #img = img / 4 + 0.5     # unnormalize
      for i in range(len(self.mean_norm)):
        img[i, :, :] = img[i, :, :] * self.std_norm[i] + self.mean_norm[i]
      npimg = img.cpu().numpy()
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
    image_diff = np.abs(torchvision.utils.make_grid(imgs - old_imgs).cpu().numpy())
    maximum_element = np.max(image_diff)
    image_diff/=maximum_element
    plt.imshow(np.transpose(image_diff, (1, 2, 0)))

  def all_changed(self, original_labels, predictions):
    np_orig = original_labels.cpu().numpy()
    np_preds = predictions.cpu().numpy()
    return np.all(np_orig != np_preds)

  def clamp_images(self, images):
    """Clamps image to between minimum and maximum range in place."""
    for i in range(len(self.mean_norm)):
      minimum_value = (0 - self.mean_norm[i])/self.std_norm[i]
      maximum_value = (1 - self.mean_norm[i])/self.std_norm[i]
      torch.clamp(images[:, i,:,:], min=minimum_value, max=maximum_value, out=images[:,i,:,:])

  def adversary_batch(self, data, model, target_class, image_reg, lr):
      # Load in first <batch_size> images for validation
      images, labels =  data
      images = images.cuda()
      original_labels = labels.cuda()
      # original_labels = labels.clone()
      inputs = Variable(images, requires_grad = True)
      opt = optim.SGD(test(inputs), lr=lr, momentum=0.9)
      self.clamp_images(images)
      old_images = images.clone()
      outputs = model(inputs)
      predicted = torch.max(outputs.data, 1)[1]
      if target_class == -1:
        new_labels = torch.topk(outputs, 2, 1)[1][:, 1]
      else:
        new_labels = Variable(torch.LongTensor([target_class]*self.batch_size)).cuda()
      iters = 0
      min_iters = 0
      while not self.all_changed(original_labels, predicted):
        if self.verbose:
          print "Iteration {}".format(iters)
        opt.zero_grad()
        self.clamp_images(images)
        outputs = model(inputs)
        model_loss = self.CrossEntropy(outputs, new_labels).cuda()
        image_loss = self.MSE(inputs, Variable(old_images)).cuda()
        loss = model_loss + image_reg*image_loss
        predicted = torch.max(outputs.data, 1)
        if self.verbose:
          print "Target Class Weights Minus Predicted Weights:"
          print outputs.data[:, new_labels.data][:,0] #  - predicted[0]
          print outputs.data[:, target_class] 
        predicted = predicted[1]
        iters += 1
        if self.all_changed(original_labels, predicted):
          if self.show_images:
            self.save_figure(inputs.data, "After_{}_{}".format(image_reg, lr))
            self.save_figure(old_images, "Before_{}_{}".format(image_reg, lr))
            self.diff(images, old_images)
            plt.show()
        else:
            loss.backward()
            opt.step()
            # if target_class == -1:
            #   new_labels = torch.topk(model(inputs), 2, 1)[1][:, 1]
      return iters, self.MSE(images, Variable(old_images))

  def create_one_adversary_batch(self, target_class, image_reg, lr):
      # Load pretrained network
      model = models.resnet101(pretrained=True)
      model.cuda()
      model.eval()
      for parameter in model.parameters():
          parameter.requires_grad = False
      data = next(iter(self.val_loader))
      return self.adversary_batch(data, model, target_class, image_reg, lr)


  def create_all_adversaries(self, target_class, image_reg, lr):
      # Load pretrained network
      model = models.resnet101(pretrained=True)
      model.cuda()
      model.eval()
      ave_mse = 0.0
      total_images = 0.0
      for parameter in model.parameters():
          parameter.requires_grad = False
      for iteration, batch in enumerate(self.val_loader, 1):
        total_images += self.batch_size
        iters, mse = self.adversary_batch(batch, model, target_class, image_reg, lr)
        ave_mse += mse.data.cpu().numpy()[0]
        print "At iteration {}, the average mse is {}".format(total_images, ave_mse/float(total_images))
        print "That batch took {} iterations".format(iters)
      return ave_mse/float(total_images)
      


