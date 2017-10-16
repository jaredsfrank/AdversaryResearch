"""
Runs bayesian optimization on a function

NOTE:
    Currently I am using this file to just test hyperparams for the GP Regression
    Therefore it currently takes parameters that correspond to some of the tunable hyperparameters
    Example Run:
        python bayesian_optimization.py -100 100 -100 100 1 4
        Corresponds to running with:
            GaussianLiklihood log_noise_bounds (-100, 100)
            mean_module (-100, 199)
            constant_boundslog_lengthscale_bounds (1, 4)

        python bayesian_optimization.py -5 5 -1 1 -3 5
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
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("log_noise_min", 
                    help="Minimum of Log Noise",
                    type=float)
parser.add_argument("log_noise_max", 
                    help="Maximum of Log Noise",
                    type=float)
parser.add_argument("const_min", 
                    help="Constant Bounds Minimum",
                    type=float)
parser.add_argument("const_max", 
                    help="Constant Bounds Maximum",
                    type=float)
parser.add_argument("cov_min", 
                    help="Covariance log lengthscale minimum",
                    type=float)
parser.add_argument("cov_max", 
                    help="Covariance log lengthscale maximum",
                    type=float)
args = parser.parse_args()


class ExactGPModel(gpytorch.GPModel):
    def __init__(self, log_noise_min, log_noise_max, constant_mean_min, constant_noise_max, log_lengthscale_min, log_lengthscale_max):
        # The log_noise_bounds add a random constant to the covariance matrix diagonal
        super(ExactGPModel,self).__init__(GaussianLikelihood(log_noise_bounds=(log_noise_min, log_noise_max)))
        self.mean_module = ConstantMean(constant_bounds=(constant_mean_min, constant_noise_max))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(log_lengthscale_min, log_lengthscale_max))
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)

def plot_model_and_predictions(model, train_x, train_y, plot_train_data=True):
    f, observed_ax = plt.subplots(1, 1, figsize=(8, 8))
    test_x = Variable(torch.linspace(0, 1, 51))
    observed_pred = model(test_x)

    def ax_plot(ax, rand_var, title):
        lower, upper = rand_var.confidence_region()
        # if plot_train_data:
        print (train_x, train_y)
        ax.plot(train_x.data.numpy(), train_y.data.numpy(), 'k*')
        ax.plot(test_x.data.numpy(), rand_var.mean().data.numpy(), 'b')
        ax.fill_between(test_x.data.numpy(), lower.data.numpy(), upper.data.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(title)

    ax_plot(observed_ax, observed_pred, 'Observed Values (Likelihood)')
    return f

def find_minimum(model):
    test_x = Variable(torch.linspace(0,1, 51))
    test_y = model(test_x)
    lower, upper = test_y.confidence_region()
    kappa = 2.0
    mean, std = test_y.std(), test_y.mean()
    # boundary = mean + kappa * std
    # return test_x.data.numpy()[np.argmin(boundary.data.numpy())] 
    return test_x.data.numpy()[np.argmin(lower.data.numpy())]



def train_model(train_x, train_y, a, b, c, d, e, f):
    model = ExactGPModel(a, b, c, d, e, f)
    model.condition(train_x, train_y)
    print("im here")
    model.train()
    print('now im here')
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    optimizer.n_iter = 0
    for i in range(50):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -model.marginal_log_likelihood(output, train_y)
        loss.backward()
        optimizer.n_iter += 1
        # print('Iter %d/20 - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
        #     i + 1, loss.data[0],
        #     model.covar_module.log_lengthscale.data[0, 0],
        #     model.likelihood.log_noise.data[0]
        # ))
        optimizer.step()
    return model


def evaluate_model(model, train_x, train_y):
    # Set back to eval mode
    model.eval()
    fig = plot_model_and_predictions(model, train_x, train_y)
    fig.savefig("/scratch/jsf239/bayesian_opt_viz/_{}.png".format(len(train_x)))
    plt.show()

if __name__ == '__main__':
    x_data = [.25, .5]
    for i in range(20):
        print (x_data)
        train_x = Variable(torch.Tensor(np.array(x_data)))
        ### NOTE: Normally for bayesian optimizatoin I would not simply use a grid of values 
        ### as the train_x every time. But I'm having trouble fitting this function, so I'm just using that
        # train_x = Variable(torch.linspace(-5, 5, 25))
        # train_x = Variable(torch.linspace(0, 1, 11))
        # train_y = Variable(0.5*(train_x.data**4 - 16*train_x.data**2 + 5*train_x.data))
        train_y = Variable((torch.sin(train_x.data * (2 * math.pi))))
        model = train_model(train_x, train_y, args.log_noise_min, args.log_noise_max, args.const_min, args.const_max, args.cov_min, args.cov_max)
        print("THe model is ")
        print(model(train_x).mean().data.numpy())
        evaluate_model(model, train_x, train_y)

        new_min = find_minimum(model)
        x_data.append(new_min)