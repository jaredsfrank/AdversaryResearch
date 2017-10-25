"""
Runs bayesian optimization on a function
"""


import math
import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable
from torch.autograd import Variable
from bayes_opt.helpers import UtilityFunction, acq_max


class ExactGPModel(gpytorch.GPModel):
    def __init__(self):
        super(ExactGPModel,self).__init__(GaussianLikelihood(log_noise_bounds=(-5, 5)))
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)

class BayesOpt(object):

    def __init__(self, eval_function, bounds=np.array([[0,1]]), initial_points=30):
        self.train_x = []
        self.train_y = []
        self.eval_function = eval_function
        # Right now, only supporting 0 to 1
        self.train_x = list(np.random.random_sample(initial_points))
        for x in self.train_x:
            self.train_y.append(self.eval_function(x))

    def run_bayes_opt(self, max_iters=100):
        old_new_min = None
        for _ in range(max_iters):
            train_x_var = Variable(torch.Tensor(np.array(self.train_x)))
            print("train x is {}".format(self.train_x))
            print("train y is {}".format(self.train_y))
            train_y_var = Variable(torch.Tensor(np.array(self.train_y)))
            print("did i make it here?")
            model = self.train_model(train_x_var, train_y_var)
            new_min = find_minimum(model)
            if new_min == old_new_min:
                return new_min
            old_new_min = new_min
            self.train_x.append(new_min)
            self.train_y.append(self.eval_function(new_min))

    def train_model(self, train_x, train_y):
        model = ExactGPModel()
        model.condition(train_x, train_y)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        for i in range(100):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -model.marginal_log_likelihood(output, train_y)
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()
        return model

    def find_minimum(self, model):
        test_x = Variable(torch.linspace(-5, 5, 51))
        test_y = model(test_x)
        lower, upper = test_y.confidence_region()
        kappa = 100.0
        mean, std = test_y.std(), test_y.mean()
        boundary = mean + kappa * std
        return test_x.data.numpy()[np.argmin(boundary.data.numpy())] 

    def plot_model_and_predictions(self, model, train_x, train_y, plot_train_data=True):
        f, observed_ax = plt.subplots(1, 1, figsize=(8, 8))
        test_x = Variable(torch.linspace(-5, 5, 51))
        observed_pred = model(test_x)

        def ax_plot(ax, rand_var, title):
            lower, upper = rand_var.confidence_region()
            # if plot_train_data:
            print (train_x, train_y)
            ax.plot(train_x.data.numpy(), train_y.data.numpy(), 'k*')
            ax.plot(test_x.data.numpy(), rand_var.mean().data.numpy(), 'b')
            ax.fill_between(test_x.data.numpy(), lower.data.numpy(), upper.data.numpy(), alpha=0.5)
            ax.set_ylim([-200, 200])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            ax.set_title(title)

        ax_plot(observed_ax, observed_pred, 'Observed Values (Likelihood)')
        return f
