import abc
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


class Adverarial_Base(object):

  __metaclass__ = abc.ABCMeta

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

  def generator_hack(self, inputs):
      yield inputs

  def imshow(self, img):
    """Unnormalizes image and then shows via matplotlib."""
    # Unnormalize each channel according to channel means and stds
    for i in range(len(self.mean_norm)):
      img[i, :, :] = img[i, :, :] * self.std_norm[i] + self.mean_norm[i]
    if self.cuda:
      img = img.cpu()
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
    fig = plt.imshow(np.transpose(image_diff, (1, 2, 0)))
    fig.savefig("/scratch/jsf239/diff.png".format)


  def percent_changed(self, original_labels, predictions):
    """Returns percent of original labels that were fooled."""
    if self.cuda:
      np_orig = original_labels.cpu().numpy()
      np_preds = predictions.cpu().numpy()
    else:
      np_orig = original_labels.numpy()
      np_preds = predictions.numpy()
    return np.sum(np_orig != np_preds)/float(len(np_orig))

  def all_changed(self, original_labels, predictions):
    """Returns true if all predictions are wrong given original correct labels."""
    if self.cuda:
      np_orig = original_labels.cpu().numpy()
      np_preds = predictions.cpu().numpy()
    else:
      np_orig = original_labels.numpy()
      np_preds = predictions.numpy()
    return np.all(np_orig != np_preds)

  def CE_MSE_loss(self, inputs, outputs, old_images, new_labels, image_reg):
    model_loss = self.CrossEntropy(outputs, new_labels)
    image_loss = self.MSE(inputs, Variable(old_images))
    if self.cuda:
      model_loss = model_loss.cuda()
      image_loss = image_loss.cuda()
    loss = model_loss + image_reg*image_loss
    return loss

  def clamp_images(self, images):
    """Clamps image to between minimum and maximum range in place."""
    for i in range(len(self.mean_norm)):
      minimum_value = (0 - self.mean_norm[i])/self.std_norm[i]
      maximum_value = (1 - self.mean_norm[i])/self.std_norm[i]
      torch.clamp(images[:, i,:,:], min=minimum_value, max=maximum_value, out=images[:,i,:,:])

  def target_class_tensor(self, target_class, outputs, original_labels):
    """Returns the target class tensor.
    
    If target class is -1, use the second most likely class for each
    label that is currently correctly predicted

    """
    if target_class == -1:

      predicted_classes = torch.max(outputs.data, 1)[1]
      predicting_correct_class = predicted_classes == original_labels
      second_best_class = torch.topk(outputs, 2, 1)[1][:, 1]
      # For each label in outputs that is correctly classified, replace
      # use second best class. Otherwise, stick with current prediction
      new_labels = Variable(predicted_classes).masked_scatter_(predicting_correct_class,
                                                               second_best_class)
    else:
      new_labels = Variable(torch.LongTensor([target_class]*self.batch_size))
    if self.cuda:
      new_labels = new_labels.cuda()
    return new_labels

  @abc.abstractmethod
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
      percent_chaged: Percent of the batch that was succesfully generated

    """
    # Load in first <batch_size> images for validation
    return

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
    model = models.resnet18(pretrained=True)
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
    model = models.resnet18(pretrained=True)
    if self.cuda:
      model.cuda()
    model.eval()
    ave_mse = 0.0
    ave_percent = 0.0
    # Set all model parameters to not update during training
    for parameter in model.parameters():
        parameter.requires_grad = False
    print("Starting Iterations")
    for iteration, batch in enumerate(self.val_loader, 1):
      iters, mse, percent_changed = self.adversary_batch(batch, model, 
                                                         target_class, 
                                                         image_reg, lr)
      if self.cuda:
        ave_mse += mse# .data.cpu().numpy()[0]
      else:
        ave_mse += mse# .data.numpy()[0]
      ave_percent += percent_changed
      print("After {} images, the average mse is {}".format(self.batch_size*iteration, ave_mse/float(iteration)))
      print("That batch took {} iterations".format(iters))
      print("{}% of the batch was succesfully generated".format(percent_changed*100))
      print("The average succes rate is {}%".format(ave_percent/float(iteration)*100))
    return ave_mse/float(iteration), ave_percent/float(iteration)*100
      


