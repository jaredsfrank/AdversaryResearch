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

    def __init__(self, eval_function, bounds, initial_points=30):
        self.train_x = []
        self.train_y = []
        self.dims = bounds.shape[0]
        self.min_ = bounds[0][0]
        self.max_ = bounds[0][1]
        self.eval_function = eval_function
        # Right now, only supporting 0 to 1
        self.train_x = list(np.random.random_sample((initial_points, self.dims))*(self.max_-self.min_)+self.min_)
        for x in self.train_x:
            self.train_y += list(self.eval_function(x).data.cpu().numpy())

    def run_bayes_opt(self, max_iters=4):
        old_new_min = None
        for _ in range(max_iters):
            train_x_var = Variable(torch.Tensor(np.array(self.train_x)).cuda())
            y_range = np.max(self.train_y) - np.min(self.train_y)
            train_y_var = Variable(torch.Tensor(5/y_range*(np.array(self.train_y) - np.mean(self.train_y))).cuda())
            model = self.train_model(train_x_var, train_y_var)
            model.eval()
            new_min = self.find_minimum(model)
            print("in here, the new min is {}".format(new_min))
            if new_min == old_new_min:
                return new_min
            old_new_min = new_min
            self.train_x.append(new_min)
            self.train_y += list(self.eval_function(new_min).data.cpu().numpy())
            self.plot_model_and_predictions(model, train_x_var, train_y_var)
            plt.show()
        return old_new_min

    def train_model(self, train_x, train_y):
        print("start")
        model = ExactGPModel().cuda()
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
        test_x = Variable(torch.linspace(self.min_, self.max_, 1000).cuda())
        test_y = model(test_x)
        lower, upper = test_y.confidence_region()
        kappa = 100.0
        mean, std = test_y.std(), test_y.mean()
        boundary = mean + kappa * std
        return test_x.data.cpu().numpy()[np.argmin(boundary.data.cpu().numpy())] 

    def plot_model_and_predictions(self, model, train_x, train_y, plot_train_data=True):
        f, observed_ax = plt.subplots(1, 1, figsize=(8, 8))
        test_x = Variable(torch.linspace(self.min_, self.max_, 100).cuda())
        observed_pred = model(test_x)

        def ax_plot(ax, rand_var, title):
            lower, upper = rand_var.confidence_region()
            ax.plot(train_x.data.cpu().numpy(), train_y.data.cpu().numpy(), 'k*')
            ax.plot(test_x.data.cpu().numpy(), rand_var.mean().data.cpu().numpy(), 'b')
            ax.fill_between(test_x.data.cpu().numpy(), lower.data.cpu().numpy(), upper.data.cpu().numpy(), alpha=0.5)
            ax.set_ylim([-10, 10])
            ax.set_xlim([self.min_, self.max_])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            ax.set_title(title)

        ax_plot(observed_ax, observed_pred, 'Observed Values (Likelihood)')
        return f
