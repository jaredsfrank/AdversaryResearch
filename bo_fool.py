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
from bayesian_optimization import BayesOpt



class BOFool(adversaries.Adverarial_Base):

    
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
    np_imgs = images.numpy() 
    if self.cuda:
      images = images.cuda()
      original_labels = original_labels.cuda()
    inputs = Variable(images, requires_grad = True)
    outputs = model(inputs)
    old_images = images.clone()
    # Set target variables for model loss
    new_labels = self.target_class_tensor(target_class, outputs, original_labels)

    # Loop through each pimage, each channel, x, and y, perform bayesian optimazation
    for img_num in range(np_imgs.shape[0]):
      np_img = np_imgs[img_num]
      old_img = old_images[img_num]
      for x in range(np_img.shape[1]):
        for y in range(np_img.shape[2]):
          for c in range(np_img.shape[0]):
            def eval_function(new_val):
              img_clone = np_img.copy()
              img_clone[c, x, y] = new_val
              # print(img_clone)
              # print(img_clone.shape)
              var_img = Variable(torch.Tensor(np.expand_dims(img_clone,0)).cuda())
              img_pred = model(var_img)
              loss = self.CE_MSE_loss(var_img, img_pred, old_img, new_labels[img_num], image_reg)
              return loss
            bo = BayesOpt(eval_function)
            new_val = bo.run_bayes_opt()
            print("The new value is ".format(new_val))
            # Very unsure about whether this change will carry over
            np_imgs[img_num, c, x, y] = new_val

    model = self.make_eval_model()
    inputsw = Variable(torch.Tensor(np_imgs).cuda())
    outputs = model(inputs)
    predicted_loss, predicted_classes = torch.max(outputs.data, 1)
    max_diff = np.mean(((images - old_images).cpu().numpy().reshape(images.shape[0],-1).max(1)))
    return 1, max_diff, self.percent_changed(original_labels, predicted_classes)


