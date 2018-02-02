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

class FGSM(adversaries.Adverarial_Base):

  def make_eval_model(self):
    model = models.resnet101(pretrained=True)
    if self.cuda:
      model.cuda()
    model.eval()
    # Set all model parameters to not update during training
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model

    
  def adversary_batch(self, data, model, target_class, image_reg, lr):
    """Creates adversarial examples for one batch of data.

    Helper function for create_one_adversary_batch.

    Args:
      data: images, labels tuple of batch to be altered
      model: trained pytorch imagenet model
      target_class: int, class to target for adverserial examples
        If target_class is -1, optimize non targeted attack. Choose next closest class.
      image_reg: Regularization constant for image loss component of loss function

    Returns:
      iters: Number of iterations it took to create adversarial example
      MSE: Means Squared error between original and altered image.

    """
    # Load in first <batch_size> images for validation
    images, original_labels =  data
    targets = torch.Tensor(target_class * np.ones_like(original_labels.numpy()))
    if self.cuda:
      images = images.cuda()
      original_labels = original_labels.cuda()
    inputs = Variable(images, requires_grad = True)
    outputs = model(inputs)
    old_images = images.clone()
    # Set target variables for model loss
    new_labels = self.target_class_tensor(target_class, outputs, original_labels)
    # Clamp loss so that all pixels are in valid range (Between 0 and 1 unnormalized)
    # Compute full loss of adversarial example
    loss = self.CE_MSE_loss(inputs, outputs, old_images, new_labels, image_reg)
    if self.show_images:
      self.save_figure(inputs.data, "After_{}_{}".format(image_reg, lr))
      self.save_figure(old_images, "Before_{}_{}".format(image_reg, lr))
      self.diff(images, old_images)
      plt.show()
    loss.backward()
    inputs.data -= lr*torch.sign(inputs.grad).data
    self.clamp_images(images)

    model = self.make_eval_model()
    # outputs = model(inputs)
    # predicted_loss, predicted_classes = torch.max(outputs.data, 1)
    # max_diff = np.mean(((images - old_images).cpu().numpy().reshape(images.shape[0],-1).max(1)))

    return inputs.clone()


