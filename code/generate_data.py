from code.utils import generate_stdp_dataset
import numpy as np
import pickle
from tqdm import tqdm
from code.utils import lasso_gradient

# np.random.seed(0)
# num_examples = 100
# dims = np.linspace(100, 750, 25)
num_samples = 10

# data_dict = dict()
# for dim in tqdm(dims):
#     dim = int(dim)
#     data_dict[dim] = []
#     for _ in range(num_samples):
#         X, Y, beta_true = generate_stdp_dataset(dim, num_examples, -20, 20)
#         lmd = np.linalg.norm(X[:-1].T @ lasso_gradient(np.zeros(num_examples), Y[:-1]), ord=np.inf)/20
#         data_dict[dim].append((X, Y, beta_true, lmd))


data_dict = dict()

data_dict[40] = []
for _ in range(num_samples):
    X, Y, beta_true = generate_stdp_dataset(80, 20, -20, 20)
    lmd = np.linalg.norm(X[:-1].T @ lasso_gradient(np.zeros(20), Y[:-1]), ord=np.inf)/20
    data_dict[40].append((X, Y, None, lmd))
with open("data/synthetic_small.pkl", "wb") as f:
    pickle.dump(data_dict, f)
