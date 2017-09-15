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
    """ Initializes useful default settings and loss functions."""
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
    self.cuda = False

  def imshow(self, img):
    """Unnormalizes image and then shows via matplotlib."""
    # Unnormalize each channel according to channel means and stds
    for i in range(len(self.mean_norm)):
      img[i, :, :] = img[i, :, :] * self.std_norm[i] + self.mean_norm[i]
    if self.cuda:
      imf = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

  def save_figure(self, imgs, name):
    """Unnormalizes image, shows it, and saves it into scratch."""
    fig = plt.figure(name)
    self.imshow(torchvision.utils.make_grid(imgs))
    fig.savefig("/scratch/jsf239/{}.png".format(name))

  def load_data(self, path, batch_size = 100, shuffle = True):
    """Initializes data loader given a batch size."""
    normalize = transforms.Normalize(self.mean_norm, self.std_norm)
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
    """Visualizes difference between altered and original image."""
    fig = plt.figure("diff")
    image_diff = torchvision.utils.make_grid(imgs - old_imgs)
    if self.cuda:
      image_diff = image_diff.cpu()
    image_diff = np.abs(image_diff.numpy())
    maximum_element = np.max(image_diff)
    image_diff/=maximum_element
    plt.imshow(np.transpose(image_diff, (1, 2, 0)))

  def all_changed(self, original_labels, predictions):
    """Returns true if all predictions are wrong given original correct labels."""
    if self.cuda:
      np_orig = original_labels.cpu().numpy()
      np_preds = predictions.cpu().numpy()
    else:
      np_orig = original_labels.numpy()
      np_preds = predictions.numpy()
    return np.all(np_orig != np_preds)

  def clamp_images(self, images):
    """Clamps image to between minimum and maximum range in place."""
    for i in range(len(self.mean_norm)):
      minimum_value = (0 - self.mean_norm[i])/self.std_norm[i]
      maximum_value = (1 - self.mean_norm[i])/self.std_norm[i]
      torch.clamp(images[:, i,:,:], min=minimum_value, max=maximum_value, out=images[:,i,:,:])

  def adversary_batch(self, data, model, target_class, image_reg, lr):
    """Creates adversarial examples for one batch of data.

    Helper function for create_one_adversary_batch.

    Args:
      data: images, labels tuple of batch to be altered
      model: trained pytorch imagenet model
      target_class: int, class to target for adverserial examples
        If target_class is -1, optimize non targeted attack. Choose next closest class.
      image_reg: Regularization constant for image loss component of loss function
      lr: float, Learning rate

    Returns:
      iters: Number of iterations it took to create adversarial example
      MSE: Means Squared error between original and altered image.

    """
    # Load in first <batch_size> images for validation
    images, original_labels =  data
    if self.cuda:
      images = images.cuda()
      original_labels = original_labels.cuda()
    inputs = Variable(images, requires_grad = True)
    opt = optim.SGD(test(inputs), lr=lr, momentum=0.9)
    self.clamp_images(images)
    old_images = images.clone()
    outputs = model(inputs)
    predicted_classes = torch.max(outputs.data, 1)[1]
    # Set target variables for model loss
    if target_class == -1:
      new_labels = torch.topk(outputs, 2, 1)[1][:, 1]
    else:
      # new_labels = Variable(torch.LongTensor([target_class]*self.batch_size))
      new_labels = Variable(torch.LongTensor(np.arange(self.batch_size)))
      if self.cuda:
        new_labels = new_labels.cuda()
    iters = 0
    print new_labels
    print torch.topk(outputs, 2, 1)
    print torch.topk(outputs, 2, 1)[1][:, 1]
    return
    while not self.all_changed(original_labels, predicted_classes):
      if self.verbose:
        print "Iteration {}".format(iters)
      opt.zero_grad()
      # Clamp loss so that all pixels are in valid range (Between 0 and 1 unnormalized)
      self.clamp_images(images)
      outputs = model(inputs)
      # Compute full loss of adversarial example
      model_loss = self.CrossEntropy(outputs, new_labels)
      image_loss = self.MSE(inputs, Variable(old_images))
      if self.cuda:
        model_loss = model_loss.cuda()
        image_loss = image_loss.cuda()
      loss = model_loss + image_reg*image_loss
      predicted_loss, predicted_classes = torch.max(outputs.data, 1)
      if self.verbose:
        print "Target Class Weights Minus Predicted Weights:"
        print outputs.data[:, new_labels.data][:,0] - predicted_loss
      iters += 1
      if self.all_changed(original_labels, predicted_classes):
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
    """Create adversarial example for one random batch of data.
  
    Args:
      target_class: int, class to target for adverserial examples
        If target_class is -1, optimize non targeted attack. Choose next closest class.
      image_reg: Regularization constant for image loss component of loss function
      lr: float, Learning rate

    Returns:
      iters: Number of iterations it took to create adversarial example
      MSE: Means Squared error between original and altered image.

    """
    # Load pretrained network
    model = models.resnet101(pretrained=True)
    if self.cuda:
      model.cuda()
    model.eval()
    # Set all model parameters to not update during training
    for parameter in model.parameters():
        parameter.requires_grad = False
    data = next(iter(self.val_loader))
    return self.adversary_batch(data, model, target_class, image_reg, lr)


  def create_all_adversaries(self, target_class, image_reg, lr):
    """Create adversarial example for every image in evaluation set.
  
    Args:
      target_class: int, class to target for adverserial examples
        If target_class is -1, optimize non targeted attack. Choose next closest class.
      image_reg: Regularization constant for image loss component of loss function
      lr: float, Learning rate

    Returns:
      ave_mse: Float, average mean square error of generated adversarial examples.
    """
    # Load pretrained network
    model = models.resnet101(pretrained=True)
    if self.cuda:
      model.cuda()
    model.eval()
    ave_mse = 0.0
    total_images = 0.0
    # Set all model parameters to not update during training
    for parameter in model.parameters():
        parameter.requires_grad = False
    for iteration, batch in enumerate(self.val_loader, 1):
      total_images += self.batch_size
      iters, mse = self.adversary_batch(batch, model, target_class, image_reg, lr)
      if self.cuda:
        ave_mse += mse.data.cpu().numpy()[0]
      else:
        ave_mse += mse.data.numpy()[0]
      print "At iteration {}, the average mse is {}".format(total_images, ave_mse/float(total_images))
      print "That batch took {} iterations".format(iters)
    return ave_mse/float(total_images)
      


