import abc
import adversaries
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

WINDOW_SIZE = 15

class P_LBFGS(adversaries.Adverarial_Base):

  def __init__(self, batch_size):
    self.max_iters = 200
    adversaries.Adverarial_Base.__init__(self, batch_size)

  def check_iters(self, iters):
    """Returns true if iters has not exceed the max number of iters."""
    if self.max_iters >= 0:
      return iters < self.max_iters
    else:
      return True

  def window_image(self, old_images, images, root_x, root_y, window_size):
    """Resores all values in images besides window"""
    y_indices = np.tile(np.arange(window_size)+root_y, window_size)
    x_indices = np.repeat(np.arange(window_size)+root_x, window_size)
    mask = torch.ByteTensor(images.shape)+1
    mask[:,:,:,:] = 1
    mask[:,:,y_indices,x_indices] = 0
    images.masked_scatter_(mask, old_images)
    return images
    
  def adversary_batch(self, data, model, target_class, image_reg, lr):
    """Creates adversarial examples for one batch of data.

    Generates adversarial batch using LBFGS iterative method.

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
    opt = optim.SGD(self.generator_hack(inputs), lr=lr, momentum=0.9)
    print("learning rate is {}".format(lr))
    self.clamp_images(images)
    old_images = images.clone()
    outputs = model(inputs)
    predicted_classes = torch.max(outputs.data, 1)[1]
    original_predictions = predicted_classes.clone()
    # if self.verbose:
    print("The predicted classes are:")
    print(predicted_classes)
    print(original_labels)
    iters = 0
    # Set target variables for model loss
    new_labels = self.target_class_tensor(target_class, outputs, original_labels)
    if not self.all_changed(original_labels, predicted_classes):
      for root_x in range(images.shape[2]-WINDOW_SIZE):
        for root_y in range(images.shape[3]-WINDOW_SIZE):
          # root_x = 5
          # root_y = 5
          print("starting? {} {}".format(root_x, root_y))
          images[:] = old_images[:]
          iters = 0
          predicted_classes = original_predictions
          while self.check_iters(iters) and not self.all_changed(original_labels, predicted_classes):
            if self.verbose and iters % 20 == 0:
              print("Iteration {}".format(iters))
            opt.zero_grad()
            # Clamp loss so that all pixels are in valid range (Between 0 and 1 unnormalized)
            self.clamp_images(images)
            outputs = model(inputs)
            # Compute full loss of adversarial example
            loss = self.CE_MSE_loss(inputs, outputs, old_images, new_labels, image_reg)
            predicted_loss, predicted_classes = torch.max(outputs.data, 1)
            if self.verbose:
              print("Target Class Weights Minus Predicted Weights:")
              print(outputs.data[:, new_labels.data][:,0] - predicted_loss)
            iters += 1

            y_indices = torch.LongTensor(np.tile(np.arange(WINDOW_SIZE)+root_y, WINDOW_SIZE)).cuda()
            x_indices = torch.LongTensor(np.repeat(np.arange(WINDOW_SIZE)+root_x, WINDOW_SIZE)).cuda()
            other_images = old_images.clone()

            mask = torch.ByteTensor(images.shape)+1
            if self.cuda:
              mask = mask.cuda()
            mask[:,:,:,:] = 1
            other_images[:,:,y_indices,x_indices] = images[:,:,y_indices,x_indices] 
            images[:] = other_images[:]



            # images = self.window_image(old_images, images, root_x, root_y, WINDOW_SIZE)
            if self.check_iters(iters) and self.all_changed(original_labels, predicted_classes):
              if self.show_images:
                self.save_figure(inputs.data, "After_{}_{}".format(image_reg, lr))
                self.save_figure(old_images, "Before_{}_{}".format(image_reg, lr))
                self.diff(images, old_images)
                plt.show()
            else:
                loss.backward()
                opt.step()
                new_labels = self.target_class_tensor(target_class, outputs, original_labels)
                # self.MSE(images, Variable(old_images))
          max_diff = np.mean(((images - old_images).cpu().numpy().reshape(images.shape[0],-1).max(1)))
          print("Max diff was {}, iters was {}".format(max_diff, iters))
      else:
        return 0, 0, 0
    return iters, max_diff, self.percent_changed(original_labels, predicted_classes)