from calendar import c
import copy
from operator import neg
from re import S
from sklearn import datasets
# from jmespath import search
import numpy as np
import scipy
import argparse 
from sklearn.datasets import make_regression
from scipy.optimize import minimize
from code.FenCon import FenCon
from code.utils import generate_stdp_dataset, lasso_loss, lasso_gradient, inv_lasso_gradient, inv_lasso_gradient, logit_grad, inv_logit_grad, assymetric_loss, assymetric_gradient, inv_assymetric_gradient, robust_loss, robust_gradient, inv_robust_gradient, prox_lasso, prox_assymetric, prox_robust, plot_path, plot_single_path, plot_again, lasso_gradient_dbeta, lasso_gradient_dy, assymetric_gradient_dbeta, assymetric_gradient_dy, robust_gradient_dbeta, robust_gradient_dy, plot_really_again, ja_inv_lasso_gradient, ja_inv_assymetric_gradient, ja_inv_robust_gradient, plot_ranks
from code.discretized_z_path import WarmDiscretePass
from code.generalized_z_path import ZPath
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dim", type=int, default=160)
	parser.add_argument("--num-examples", type=int, default=80)
	parser.add_argument("--max-val", type=int, default=20)
	parser.add_argument("--min-val", type=int, default=-20)
	parser.add_argument("--loss-type", choices=["assymetric", "lasso", "robust"])
	parser.add_argument("--dataset")
	args = parser.parse_args()
	if args.dataset == "diabetes":
		diabetes = datasets.load_diabetes()
		X, Y = diabetes.data, diabetes.target
		X= X/X.max(axis=0)
		Y = (Y - Y[:-1].mean())/Y[:-1].std()
	elif args.dataset == "friedman1":
		friedman = datasets.make_friedman1()
		X, Y = friedman[0], friedman[1]
		X= X/X.max(axis=0)
		Y = (Y - Y[:-1].mean())/Y[:-1].std()
	elif args.dataset == "friedman2":
		friedman = datasets.make_friedman2()
		X, Y = friedman[0], friedman[1]
		X= X/X.max(axis=0)
		Y = (Y - Y[:-1].mean())/Y[:-1].std()
	elif args.dataset == "synthetic":	
		X, Y, beta_true = generate_stdp_dataset(args.dim, args.num_examples, args.min_val, args.max_val)
	lmd = np.linalg.norm(X[:-1].T @ lasso_gradient(np.zeros(len(X) - 1), Y[:-1]), ord=np.inf)/20

	if args.loss_type == "assymetric":
		assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-10, lmd=lmd, ja_inverse_model=ja_inv_assymetric_gradient)
		our_ranks, our_points, betas, cands = assymetric_cp.solve_CP(X, Y[:-1])
		d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, d_ranks = WarmDiscretePass(X, Y[:-1], prox_assymetric, assymetric_cp.duality_gap, lmd, tol=1e-9)
	elif args.loss_type == "robust":
		robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, lmd=lmd, ja_inverse_model=ja_inv_robust_gradient,)
		our_ranks, our_points, betas, cands = robust_cp.solve_CP(X, Y[:-1])
		d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, d_ranks = WarmDiscretePass(X, Y[:-1], prox_robust, robust_cp.duality_gap, lmd, tol=1e-10)
	elif args.loss_type == "lasso":
		lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-9, lmd=lmd, ja_inverse_model=ja_inv_lasso_gradient)
		our_ranks, our_points, betas, cands = lasso_cp.solve_CP(X, Y[:-1])
		d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, d_ranks = WarmDiscretePass(X, Y[:-1], prox_lasso, lasso_cp.duality_gap, lmd, tol=1e-8)


	

	condensed_search_space = []
	condensed_transitions = []


	for i in range(1, len(search_space)):
		if d_ranks[i] != d_ranks[i-1]:
			condensed_search_space.append(search_space[i])
			condensed_transitions.append((d_ranks[i-1], d_ranks[i]))

	condensed_search_space = list(reversed(condensed_search_space))
	condensed_transitions = list(reversed(condensed_transitions))

	# import pickle
	# with open("results/cp_ranks_lasso_friedman1.pkl","wb") as f:
	# 	pickle.dump((d_ranks, search_space, our_ranks, our_points), f)
	# breakpoint()

	plot_ranks(d_ranks, search_space, our_ranks, our_points, name="{}_{}_rank".format(args.loss_type, args.dataset))
	plot_path(d_betas, search_space, betas, cands, 'candidates', name="{}_{}".format(args.loss_type, 'z'))