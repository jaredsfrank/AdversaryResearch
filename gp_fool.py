import abc
import adversaries
import torch
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import gpytorch
from torch import optim
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable


class SpectralMixtureGPModel(gpytorch.GPModel):
    def __init__(self):
        super(SpectralMixtureGPModel,self).__init__(GaussianLikelihood(log_noise_bounds=(-5, 5)))
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = SpectralMixtureKernel(n_mixtures=3)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class GPFool(adversaries.Adverarial_Base):

  # NOTE: Currently only works for batch sizes of 1
  def adversary_batch(self, data, model, target_class, image_reg, lr):
    # Load in first <batch_size> images for validation
    images, original_labels =  data
    if self.cuda:
      images = images.cuda()
      original_labels = original_labels.cuda()
    inputs = Variable(images, requires_grad = True)
    outputs = model(inputs)
    old_images = images.clone()
    print(old_images)
    return
    # Set target variables for model loss
    new_labels = self.target_class_tensor(target_class, outputs, original_labels)
    # Clamp loss so that all pixels are in valid range (Between 0 and 1 unnormalized)
    # Compute full loss of adversarial example
    loss = self.CE_MSE_loss(inputs, outputs, old_images, new_labels, image_reg)
    gp_model = SpectralMixtureGPModel()
    gp_model.condition(train_x, loss)
    self.clamp_images(images)
    outputs = model(inputs)
    predicted_loss, predicted_classes = torch.max(outputs.data, 1)
    max_diff = np.mean(((images - old_images).cpu().numpy().reshape(images.shape[0],-1).max(1)))
    return 1, max_diff, self.percent_changed(original_labels, predicted_classes)