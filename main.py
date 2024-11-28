import argparse
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import gpytorch
import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="linear_operator")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 512))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(512, 256))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(256, 128))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(128, 2))

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
    parser.add_argument('--missing_ratio',type=float, default=0.3, help='ratio of sensor-free nodes')

    args = parser.parse_args()

    if args.dataset == 'PeMS7_228':
        num_days = 44
        data = pd.read_csv("datasets/PeMS7_228/PeMSD7_V_228.csv", header=None).values
    elif args.dataset == 'PeMS7_1026':
        num_days = 44
        data = pd.read_csv("datasets/PeMS7_1026/PeMSD7_V_1026.csv", header=None).values
    elif args.dataset == 'Seattle':
        num_days = 365
        data = pd.read_pickle('./dataset/Seattle/speed_matrix_2015').values # (D*L_d, K)

    test_days = 2
    standardizer = StandardScaler()
    standardizer.fit(data)
    data = standardizer.transform(data)
    data = torch.Tensor(data) # data shape (D*L_d, K

    # Set the random seed
    np.random.seed(random_seed)
    # Get the number of columns in the data
    num_columns = data.shape[1]
    # Create a numpy array of all column indices
    all_indices = np.arange(num_columns)
    # Calculate the number of columns to be selected
    num_selected_columns = int(num_columns * args.missing_ratio)
    # Randomly select column indices
    selected_indices = np.random.choice(num_columns, num_selected_columns, replace=False)
    # Find the indices that are in all_indices but not in selected_indices
    rest_indices = np.setdiff1d(all_indices, selected_indices)

    X = data[:, rest_indices]
    Y = data[:, selected_indices]
    train_X = X[:int((num_days-test_days)*288),:].contiguous()
    train_Y = Y[:int((num_days-test_days)*288),:].contiguous()
    test_X = X[int((num_days-test_days)*288):,:].contiguous()
    test_Y = Y[int((num_days-test_days)*288):,:].contiguous()

    train_x = train_X
    train_y = train_Y[:,0]
    test_x = test_X
    test_y = test_Y[:,0]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        train_x, train_y, test_x, test_y = train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_X, train_y, likelihood, train_x.shape[1])

    if torch.cuda.is_available():
        model = model.to(device)
        likelihood = likelihood.to(device)

    training_iterations = 240

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

        if (i + 1) % 5 == 0:
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
                preds = model(test_x)
                # Convert tensors to numpy arrays and reshape them to 2D for the inverse_transform method
                preds_np = preds.mean.cpu().numpy().reshape(-1, 1)
                test_y_np = test_y.cpu().numpy().reshape(-1, 1)
                expanded_preds_np = np.repeat(preds_np, num_columns, axis=1)
                expanded_test_y_np = np.repeat(test_y_np, num_columns, axis=1)

                # Apply inverse_transform
                preds_np = standardizer.inverse_transform(expanded_preds_np)[:,-1]
                test_y_np = standardizer.inverse_transform(expanded_test_y_np)[:,-1]
                # Compute MAE
                mae = np.mean(np.abs(preds_np - test_y_np))
                # Compute RMSE
                rmse = np.sqrt(np.mean((preds_np - test_y_np)**2))
                # Compute MAPE
                mape = np.mean(np.abs((test_y_np - preds_np) / test_y_np))

                # Print MAE, RMSE, and MAPE in one line
                print('Iteration {}, Test MAE: {}, RMSE: {}, MAPE: {}%'.format(i+1, mae, rmse, mape))
            model.train()