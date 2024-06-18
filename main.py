import argparse
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import gpytorch
import tqdm

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, data_dim):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = FeatureExtractor(data_dim)

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == '__main__':

    random_seed = 42
    
    parser = argparse.ArgumentParser(description='ST-VGP')
    parser.add_argument('--dataset', type=str, default='PeMS7_228', help='dataset name')

    args = parser.parse_args()

    if args.dataset == 'PeMS7_228':
        data = pd.read_csv("datasets/PeMS7_228/PeMSD7_V_228.csv", header=None).values
    elif args.dataset == 'PeMS7_1026':
        data = pd.read_csv("datasets/PeMS7_1026/PeMSD7_V_1026.csv", header=None).values
    elif args.dataset == 'Seattle':
        data = pd.read_pickle('./dataset/Seattle/speed_matrix_2015').values # (D*L_d, K)

    standardizer = StandardScaler()
    standardizer.fit(data)
    data = standardizer.transform(data)
    data = torch.Tensor(data) # data shape (D*L_d, K)

    


    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    training_iterations = 60

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def train():
        iterator = tqdm.tqdm(range(training_iterations))
        for i in iterator:
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            loss.backward()
            iterator.set_postfix(loss=loss.item())
            optimizer.step()
