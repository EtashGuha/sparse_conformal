import imp
import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pandas as pd
import time
from code.utils import generate_stdp_dataset, lasso_loss, lasso_gradient, inv_lasso_gradient, inv_lasso_gradient, logit_grad, inv_logit_grad, assymetric_loss, assymetric_gradient, inv_assymetric_gradient, robust_loss, robust_gradient, inv_robust_gradient, prox_lasso, prox_assymetric, prox_robust, plot_path, plot_single_path, plot_again, lasso_gradient_dbeta, lasso_gradient_dy, assymetric_gradient_dbeta, assymetric_gradient_dy, robust_gradient_dbeta, robust_gradient_dy, plot_really_again, ja_inv_lasso_gradient, ja_inv_assymetric_gradient, ja_inv_robust_gradient
import copy

random_state = 414

def calc_full_homo(X, Y, lmd, dual_gap_func, epsilon=1e-3, prox_func=prox_lasso):
	tic = time.time()
	eps_ = epsilon / np.linalg.norm(Y[:-1]) ** 2
	alpha = [lmd/ X[:-1].shape[0]]
	coef = lasso_path(X[:-1], Y[:-1], alphas=alpha, tol=eps_)[1].ravel()
	coef_c = coef.copy()
	tic = time.time()
	coefs, y_s = homotopy_path(X, Y, lmd, coef_c, y_t=max(Y), dual_gap_func=dual_gap_func, epsilon=eps_, prox_func=prox_func)
	t_homotopy = time.time() - tic
	return coefs, y_s

def homotopy_path(X, Y, lambda_, coef, y_t, dual_gap_func, epsilon=1e-8, nu=1., prox_func=prox_lasso):

	eps_0 = epsilon / 10.
	step_size = np.sqrt(2. * (epsilon - eps_0) / nu)
	
	Y_t = np.array(list(Y[:-1]) + [y_t], order='F')
	y_stop = min(Y[:-1])

	coefs = []
	y_s = []

	while y_t < y_stop:

		y_t = min(y_t + step_size, y_stop)
		Y_t[-1] = y_t
		tol = eps_0 / np.linalg.norm(Y_t) ** 2
		# alpha = [lambda_ / X.shape[0]]
		coef,  gap, theta_estim = prox_func(X, Y_t, lambda_, dual_gap_func, initial_beta=copy.deepcopy(coef), tol=tol, max_iter=10000)
		coefs.append(copy.deepcopy(coef))
		y_s.append(y_t)
	while y_t > y_stop:
		y_t = max(y_t - step_size, y_stop)
		Y_t[-1] = y_t
		tol = eps_0 / np.linalg.norm(Y_t) ** 2
		# alpha = [lambda_ / X.shape[0]]
		coef, gap, theta_estim = prox_func(X, Y_t, lambda_, dual_gap_func, initial_beta=copy.deepcopy(coef), tol=tol, max_iter=10000)
		coefs.append(copy.deepcopy(coef))
		y_s.append(y_t)
	return coefs, y_s


def calc_homotopy_cp(X, Y, alpha, lambda_, dual_gap_func, epsilon=1e-8, nu=1., prox_func=prox_lasso):
	coefs, y_s = calc_full_homo(X, Y, lambda_, dual_gap_func, epsilon=epsilon, prox_func=prox_func)

	bounds = []
	in_interval = False

	num_examples = len(Y)
	num_greaters = []
	for idx in range(len(coefs)):
		coef = coefs[idx]
		tmp_y = copy.deepcopy(Y)
		tmp_y[-1] = y_s[idx]
		residuals = abs(tmp_y  - X @ coef)
		num_greater = len(np.where(residuals > residuals[-1])[0])
		num_greaters.append(num_greater)
		if 1/num_examples * num_greater > alpha:
			bounds.append(y_s[idx])
	if len(bounds) == 0:
		return 0, 0
	return min(bounds), max(bounds)


