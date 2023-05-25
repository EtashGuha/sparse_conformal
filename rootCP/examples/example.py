from rootCP.rootcp.models import FenConCP
import numpy as np
from sklearn.datasets import make_regression
from rootCP.rootcp import rootCP, models
from sklearn.linear_model import Lasso
from code.generalized_z_path import ZPath
from code.utils import generate_stdp_dataset, lasso_loss, lasso_gradient, inv_lasso_gradient, inv_lasso_gradient, logit_grad, inv_logit_grad, assymetric_loss, assymetric_gradient, inv_assymetric_gradient, robust_loss, robust_gradient, inv_robust_gradient, prox_lasso, prox_assymetric, prox_robust, plot_path, plot_single_path, lasso_gradient_dbeta, lasso_gradient_dy, assymetric_gradient_dbeta, assymetric_gradient_dy, robust_gradient_dbeta, robust_gradient_dy
from code.discretized_z_path import WarmDiscretePass

import time
import warnings
warnings.filterwarnings("ignore")
n_samples, n_features = (200, 100)
# X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=1)
# X /= np.linalg.norm(X, axis=0)
# y = (y - y.mean()) / y.std()
X, y, beta_true = generate_stdp_dataset(n_features, n_samples, -20, 20)
print("The target is", y[-1])

lmd = np.linalg.norm(X.T @ lasso_gradient(np.zeros(len(X)), y), ord=np.inf)/20
# assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, lmd=lmd)
# cands, betas = assymetric_cp.solve(X, y[:-1])
# d_betas, d_gaps, d_kinks, d_kink_betas, search_space = WarmDiscretePass(X, y[:-1], prox_assymetric, assymetric_cp.duality_gap, lmd)
# plot_path(d_betas, search_space, betas, cands, 'candidates', name="{}_{}".format("assymetric", 'z'))

# t1 = time.time()
# assymetric_model = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, lmd=lmd)
# assymetric_cp = models.FenConCP(model = assymetric_model, name="assymetric_cp")
# cp = rootCP.conformalset(X, y[:-1], assymetric_cp)
# t2 = time.time()
# print("Assymetric CP set is {} and time {}".format(cp, t2-t1))

# t1 = time.time()
# robust_model = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, lmd=lmd)
# robust_cp = models.FenConCP(model = robust_model, name="robust_cp")
# cp = rootCP.conformalset(X, y[:-1], robust_cp)
# t2 = time.time()
# print("Robust CP set is {} and time {}".format(cp, t2-t1))

t1 = time.time()
lasso_model = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, lmd=lmd)
lasso_cp = models.FenConCP(model = lasso_model, name="lasso_cp")
cp = rootCP.conformalset(X, y[:-1], lasso_cp)
t2 = time.time()
print("Lasso CP set is {} and time {}".format(cp, t2-t1))

# lmd = 0.5
t1 = time.time()
ridge_regressor = models.ridge(lmd=lmd)
cp = rootCP.conformalset(X, y[:-1], ridge_regressor)
t2 = time.time()
print("Ridge CP set is {} and time {}".format(cp, t2-t1))


t1 = time.time()
lmd = np.linalg.norm(X.T.dot(y), ord=np.inf) / 30
model = Lasso(alpha=lmd / X.shape[0], warm_start=False)
regression_model = models.regressor(model=model)
cp = rootCP.conformalset(X, y[:-1], regression_model)
t2 = time.time()
print("Lasso CP set is{} and time {}".format(cp, t2-t1))
