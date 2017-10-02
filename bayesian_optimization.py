import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from torch import optim
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable
from torch.autograd import Variable


class ExactGPModel(gpytorch.GPModel):
    def __init__(self):
        super(ExactGPModel,self).__init__(GaussianLikelihood(log_noise_bounds=(-5, 5)))
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)

def plot_model_and_predictions(model, plot_train_data=True):
    f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
    test_x = Variable(torch.linspace(0, 1, 51))
    observed_pred = model(test_x)

    def ax_plot(ax, rand_var, title):
        lower, upper = rand_var.confidence_region()
        if plot_train_data:
            ax.plot(train_x.data.numpy(), train_y.data.numpy(), 'k*')
        ax.plot(test_x.data.numpy(), rand_var.mean().data.numpy(), 'b')
        ax.fill_between(test_x.data.numpy(), lower.data.numpy(), upper.data.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(title)

    ax_plot(observed_ax, observed_pred, 'Observed Values (Likelihood)')
    return f

def find_minimum(model):
	test_x = Variable(torch.linspace(0, 1, 51))
	test_y = model(test_x)
	lower, upper = test_y.confidence_region()
	print(lower)


def train_model(train_x, train_y):
	model = ExactGPModel()
	model.condition(train_x, train_y)
	fig = plot_model_and_predictions(model, plot_train_data=False)
	plt.show()
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=0.1)
	optimizer.n_iter = 0
	for i in range(50):
	    optimizer.zero_grad()
	    output = model(train_x)
	    loss = -model.marginal_log_likelihood(output, train_y)
	    loss.backward()
	    optimizer.n_iter += 1
	    print('Iter %d/20 - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
	        i + 1, loss.data[0],
	        model.covar_module.log_lengthscale.data[0, 0],
	        model.likelihood.log_noise.data[0]
	    ))
	    optimizer.step()
	return model


def evaluate_model(model):
	# Set back to eval mode
	model.eval()
	fig = plot_model_and_predictions(model)
	plt.show()

if __name__ == '__main__':
	train_x = Variable(torch.linspace(0, 1, 11))
	train_y = Variable(torch.sin(train_x.data * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2)
	model = train_model(train_x, train_y)
	evaluate_model(model)
	find_minimum(model)
