from calendar import c
import copy
from operator import neg
from re import S
from sklearn import datasets
# from jmespath import search
import numpy as np
import scipy
import argparse 
from scipy.optimize import minimize
from code.FenCon import FenCon
from code.utils import generate_stdp_dataset, lasso_loss, lasso_gradient, inv_lasso_gradient, inv_lasso_gradient, logit_grad, inv_logit_grad, assymetric_loss, assymetric_gradient, inv_assymetric_gradient, robust_loss, robust_gradient, inv_robust_gradient, prox_lasso, prox_assymetric, prox_robust, plot_path, plot_single_path, plot_again, lasso_gradient_dbeta, lasso_gradient_dy, assymetric_gradient_dbeta, assymetric_gradient_dy, robust_gradient_dbeta, robust_gradient_dy, plot_really_again, ja_inv_lasso_gradient, ja_inv_assymetric_gradient, ja_inv_robust_gradient
from code.discretized_z_path import WarmDiscretePass
from code.generalized_z_path import ZPath
import matplotlib.pyplot as plt
from multiprocessing import Pool
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dim", type=int, default=100)
	parser.add_argument("--num-examples", type=int, default=100)
	parser.add_argument("--max-val", type=int, default=20)
	parser.add_argument("--min-val", type=int, default=-20)
	parser.add_argument("--loss-type", choices=["assymetric", "lasso", "robust"])

	args = parser.parse_args()

	all_gradients = []
	def run(index):
		X, Y, beta_true = generate_stdp_dataset(args.dim, args.num_examples, args.min_val, args.max_val)

		lmd = np.linalg.norm(X[:-1].T @ lasso_gradient(np.zeros(args.num_examples), Y[:-1]), ord=np.inf)/20

		search_space = np.linspace(min(Y), max(Y), 30)
		if args.loss_type == "assymetric":
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-10, lmd=lmd, ja_inverse_model=ja_inv_assymetric_gradient)
			chosen_cp = assymetric_cp
			d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_assymetric, assymetric_cp.duality_gap, lmd,  tol=1e-9)
		elif args.loss_type == "robust":
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-4, lmd=lmd, ja_inverse_model=ja_inv_robust_gradient,)
			chosen_cp = robust_cp
			d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_robust, robust_cp.duality_gap, lmd,  tol=1e-4)

		elif args.loss_type == "lasso":
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-12, bisection_tol=1e-10, lmd=lmd, ja_inverse_model=ja_inv_lasso_gradient)
			chosen_cp = lasso_cp
			d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_lasso, lasso_cp.duality_gap, lmd, tol=1e-12)

		temp_gradients = []
		for i in range(len(search_space) - 1):
			try:
				temp_gradients.append(max((d_betas[i+1] - d_betas[i])/(search_space[i+1] - search_space[i])))
			except:
				breakpoint()
		return temp_gradients

	# run(0)
	with Pool(10) as p:
		all_gradients = p.map(run, list(range(100)))
	breakpoint()
	plt.hist(np.asarray(all_gradients), bins=25)
	plt.savefig("code/official_images/lasso_grad_dist_{}d_{}ne_{}.png".format(args.dim, args.num_examples, args.loss_type))