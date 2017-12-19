# x = np.loadtxt(str(i).zfill(4)+"all_scores.csv", delimiter = ",")[:-1,:-1].T; x = np.repeat(np.repeat(x, 28, axis = 0), 28, axis = 1); plt.imshow(x, vmin = 0); plt.show()


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
from convert_label import convert_label
import torch.optim as optim

WINDOW_SIZE = 28

class P_LBFGS(adversaries.Adverarial_Base):

  def __init__(self, batch_size):
    self.max_iters = 1000
    adversaries.Adverarial_Base.__init__(self, batch_size)

  def check_iters(self, iters):
    """Returns true if iters has not exceed the max number of iters."""
    if self.max_iters >= 0:
      return iters < self.max_iters
    else:
      return True

  def window_image_old(self, old_images, images, root_x, root_y, window_size):
    """Resores all values in images besides window"""
    y_indices = np.tile(np.arange(window_size)+root_y, window_size)
    x_indices = np.repeat(np.arange(window_size)+root_x, window_size)
    mask = torch.ByteTensor(images.shape)+1
    mask[:,:,:,:] = 1
    mask[:,:,y_indices,x_indices] = 0
    images.masked_scatter_(mask, old_images)
    return images


  def window_image(self, old_images, images, root_x, root_y, window_size):
    """Resores all values in images besides window in place"""
    y_indices = torch.LongTensor(np.tile(np.arange(window_size)+root_y, window_size)).cuda()
    x_indices = torch.LongTensor(np.repeat(np.arange(window_size)+root_x, window_size)).cuda()
    other_images = old_images.clone()
    mask = torch.ByteTensor(images.shape)+1
    if self.cuda:
      mask = mask.cuda()
    mask[:,:,:,:] = 1
    other_images[:,:,y_indices,x_indices] = images[:,:,y_indices,x_indices] 
    images[:] = other_images[:]


  def create_1000_batches(self, target_class, image_reg, lr):
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
    assert(len(self.sub_dir) == 0 or self.sub_dir[-1] == "/")
    self.iteration = 0
    original_subdir = self.sub_dir
    successes = 0
    for i in range(1000):
      model = models.resnet101(pretrained=True)
      self.sub_dir = original_subdir + str(i).zfill(4)
      if self.cuda:
        model.cuda()
      model.eval()
      # Set all model parameters to not update during training
      for parameter in model.parameters():
          parameter.requires_grad = False
      data = next(iter(self.val_loader))
      _, _, _, success = self.adversary_batch(data, model, target_class, image_reg, lr)
      successes += success
      np.savetxt("/scratch/jsf239/{}succ_percent.csv".format(original_subdir), np.array([successes/float(i)]), delimiter = ",")
    
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
    self.clamp_images(images)
    old_images = images.clone()
    outputs = model(inputs)
    predicted_classes = torch.max(outputs.data, 1)[1]
    original_predictions = predicted_classes.clone()
    # if self.verbose:
    print("The predicted classes are:")
    print(convert_label(predicted_classes.cpu().numpy()))
    print(convert_label(original_labels.cpu().numpy()))

    # Set target variables for model loss
    new_labels = self.target_class_tensor(target_class, outputs, original_labels)
    all_scores = np.zeros((images.shape[0], (images.shape[2]-WINDOW_SIZE)//(WINDOW_SIZE)+1, (images.shape[3]-WINDOW_SIZE)//(WINDOW_SIZE)+1))
    success_list = np.zeros(images.shape[0])
    for root_x in range(0, images.shape[2]-WINDOW_SIZE, WINDOW_SIZE):
      for root_y in range(0, images.shape[3]-WINDOW_SIZE, WINDOW_SIZE):
        print("starting {} {}".format(root_x, root_y))
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
          iters += 1
          if self.verbose and iters % 20 == 0:
            print("Target Class Weights Minus Predicted Weights of Original:")
            print(outputs.data[:, new_labels.data][:,0] - predicted_loss)
          
          self.window_image(old_images, images, root_x, root_y, WINDOW_SIZE)

          outputs = model(inputs)
          loss = self.CE_MSE_loss(inputs, outputs, old_images, new_labels, image_reg)
          predicted_loss, predicted_classes = torch.max(outputs.data, 1)
          loss.backward()
          opt.step()
          new_labels = self.target_class_tensor(target_class, outputs, original_labels)
          all_scores[:, root_x//(WINDOW_SIZE), root_y//(WINDOW_SIZE)] += (original_labels.cpu().numpy() == predicted_classes.cpu().numpy()).astype("float64")

        # success += len(np.where(all_scores[:, root_x//(WINDOW_SIZE), root_y//(WINDOW_SIZE)]<self.max_iters)[0])
        success_list = np.maximum(success_list, all_scores[:, root_x//(WINDOW_SIZE), root_y//(WINDOW_SIZE)] < self.max_iters)
        success = np.sum(success_list)
        print("There are {} successes".format(success))
        max_diff = np.mean(((images - old_images).cpu().numpy().reshape(images.shape[0],-1).max(1)))
        print("Max diff was {}, iters was {}".format(max_diff, iters))

        if self.show_images:
          self.save_figure(inputs.data, "After_{}_{}".format(image_reg, lr))
          # self.save_figure(old_images, "Before_{}_{}".format(image_reg, lr))
          plt.show()
        print((outputs.data[:, new_labels.data][:,0] - predicted_loss).cpu().numpy())
        # np.savetxt("/scratch/jsf239/{}all_scores.csv".format(self.sub_dir), all_scores, delimiter = ",")
        np.save("/scratch/jsf239/{}all_scores".format(self.sub_dir), all_scores)
    return success, iters, max_diff, success

