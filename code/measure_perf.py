from calendar import c
import copy
from re import S
from code.approx_homotopy import calc_full_homo

# from jmespath import search
import numpy as np
import scipy
import argparse 
from sklearn.datasets import make_regression
from scipy.optimize import minimize
from code.FenCon import FenCon
from code.utils import generate_stdp_dataset, lasso_loss, lasso_gradient, inv_lasso_gradient, inv_lasso_gradient, logit_grad, inv_logit_grad, assymetric_loss, assymetric_gradient, inv_assymetric_gradient, robust_loss, robust_gradient, inv_robust_gradient, prox_lasso, prox_assymetric, prox_robust, plot_path, plot_single_path, plot_again, lasso_gradient_dbeta, lasso_gradient_dy, assymetric_gradient_dbeta, assymetric_gradient_dy, robust_gradient_dbeta, robust_gradient_dy, plot_really_again, ja_inv_lasso_gradient, ja_inv_assymetric_gradient, ja_inv_robust_gradient
from code.discretized_z_path import WarmDiscretePass
from code.generalized_z_path import ZPath
import time
import matplotlib.pyplot as plt
import sklearn

def calc_errors(betas, cands, d_betas, search_space):
	errors = []
	for idx in range(len(search_space)):
		real_cand = search_space[idx]
		true_beta = d_betas[idx]
		aset = np.where(abs(true_beta) > 1e-8)[0]
		if real_cand <= cands[0]:
			# slopes = (betas[1] - betas[0])/(cands[1] - cands[0])
			# final_beta = betas[0] + slopes * (real_cand - cands[0])
			final_beta = betas[0]
		elif real_cand >= cands[-1]:
			# slopes = (betas[-1] - betas[-2])/(cands[-1] - cands[-2])
			# final_beta = betas[-1] + slopes * (real_cand - cands[-1]) 
			final_beta = betas[-1]
		else:
			try:
				for i in range(len(cands) - 1):
						if real_cand <= cands[i + 1] and real_cand >= cands[i]:
							if np.isclose(cands[i+1], cands[i]).all():
								final_beta = betas[i]
							else:
								slopes = (betas[i + 1] - betas[i])/(cands[i+1] - cands[i])
								final_beta = betas[i] + slopes * (real_cand - cands[i]) 
								break
			except:
				breakpoint()
		errors.append(np.mean(np.linalg.norm(final_beta - true_beta, ord=np.inf)))
	return errors

def calc_duality_gaps(betas, cands, search_space, X, Y, dual_gap_func, proximal_func, primal, lmd):
	Y_z = copy.deepcopy(Y)
	errors = []
	true_gaps = []
	true_primals = []
	for idx in range(len(search_space)):
		real_cand = search_space[idx]
		if real_cand <= cands[0]:
			# slopes = (betas[1] - betas[0])/(cands[1] - cands[0])
			# final_beta = betas[0] + slopes * (real_cand - cands[0])
			final_beta = betas[0]
		elif real_cand >= cands[-1]:
			# slopes = (betas[-1] - betas[-2])/(cands[-1] - cands[-2])
			# final_beta = betas[-1] + slopes * (real_cand - cands[-1]) 
			final_beta = betas[-1]
		else:
			try:
				for i in range(len(cands) - 1):
						if real_cand <= cands[i + 1] and real_cand >= cands[i]:
							if np.isclose(cands[i+1], cands[i]).all():
								final_beta = betas[i]
							else:
								slopes = (betas[i + 1] - betas[i])/(cands[i+1] - cands[i])
								final_beta = betas[i] + slopes * (real_cand - cands[i]) 
								break
			except:
				breakpoint()
		Y_z[-1] = real_cand
		true_beta, gap, _ = proximal_func(X, Y_z, lmd, dual_gap_func, tol=1e-8, initial_beta=final_beta)
		true_gaps.append(gap)
		true_primals.append(primal(X, Y_z, beta=true_beta))
		errors.append(primal(X, Y_z, beta=final_beta))

	return np.mean(np.asarray(errors) - np.asarray(true_primals))

def get_all_primals(betas, cands, search_space, X, Y, dual_gap_func, lmd):
	Y_z = copy.deepcopy(Y)
	errors = []
	for idx in range(len(search_space)):
		real_cand = search_space[idx]
		if real_cand <= cands[0]:
			# slopes = (betas[1] - betas[0])/(cands[1] - cands[0])
			# final_beta = betas[0] + slopes * (real_cand - cands[0])
			final_beta = betas[0]
		elif real_cand >= cands[-1]:
			# slopes = (betas[-1] - betas[-2])/(cands[-1] - cands[-2])
			# final_beta = betas[-1] + slopes * (real_cand - cands[-1]) 
			final_beta = betas[-1]
		else:
			try:
				for i in range(len(cands) - 1):
						if real_cand <= cands[i + 1] and real_cand >= cands[i]:
							if np.isclose(cands[i+1], cands[i]).all():
								final_beta = betas[i]
							else:
								slopes = (betas[i + 1] - betas[i])/(cands[i+1] - cands[i])
								final_beta = betas[i] + slopes * (real_cand - cands[i]) 
								break
			except:
				breakpoint()
		Y_z[-1] = real_cand
		
		errors.append(dual_gap_func(X, Y_z, beta=final_beta))

	return errors


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--max-val", type=int, default=20)
	parser.add_argument("--min-val", type=int, default=-20)
	parser.add_argument("--loss-type", choices=["assymetric", "lasso", "robust"])
	args = parser.parse_args()

	all_errors = []
	approx_times = []
	all_times = []
	approx_errors = []
	max_asets = []
	our_gaps = []
	approx_gaps = []
	range_space = np.linspace(10, 100, num=10)
	range_space= np.linspace(10, 50, num=2)
	for dim in range_space:
		dim = int(dim)
		num_examples = int(dim/2)
		X, Y, beta_true = generate_stdp_dataset(dim, num_examples, args.min_val, args.max_val)

		lmd = np.linalg.norm(X[:-1].T @ lasso_gradient(np.zeros(num_examples), Y[:-1]), ord=np.inf)/20
		if args.loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			cands, betas, duals = assymetric_cp.solve(X, Y[:-1])
			t2 = time.time()
			coefs, ys = calc_full_homo(X, Y, lmd=lmd, dual_gap_func=assymetric_cp.duality_gap, prox_func=prox_assymetric, epsilon = 1e-4)
			t3 = time.time()
			dual_gap_func = assymetric_cp.duality_gap
			# d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals = WarmDiscretePass(X, Y[:-1], prox_assymetric, assymetric_cp.duality_gap, lmd)
		elif args.loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, lmd=lmd,mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_robust_gradient)
			cands, betas, duals = robust_cp.solve(X, Y[:-1])
			t2 = time.time()
			print("time: {}".format(t2 - t1))
			coefs, ys = calc_full_homo(X, Y, lmd=lmd, dual_gap_func=robust_cp.duality_gap, prox_func=prox_robust, epsilon =1e-10)
			t3 = time.time()
			print("time: {}".format(t3 - t2))
			dual_gap_func = robust_cp.duality_gap
			# d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals = WarmDiscretePass(X, Y[:-1], prox_robust, robust_cp.duality_gap, lmd)
		elif args.loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			cands, betas, duals = lasso_cp.solve(X, Y[:-1])
			t2 = time.time()
			# reg = sklearn.linear_model.LassoLars(alpha=lmd / X.shape[0], fit_path=False, fit_intercept=False, normalize=False, max_iter=1000000, eps=1e-14)
			# reg.fit(X, Y)
			coefs, ys = calc_full_homo(X, Y, lmd=lmd, dual_gap_func=lasso_cp.duality_gap, epsilon =1e-4)
			t3 = time.time()
			dual_gap_func = lasso_cp.duality_gap
			d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals = WarmDiscretePass(X, Y[:-1], prox_lasso, lasso_cp.duality_gap, lmd)

		cands.reverse()
		betas.reverse()

		coefs.reverse()
		ys.reverse()
		search_space = np.linspace(min(Y), max(Y), 100)
		our_gaps.append(calc_duality_gaps(betas, cands, search_space, X, Y, dual_gap_func, lmd))
		approx_gaps.append(calc_duality_gaps(coefs, ys, search_space, X, Y, dual_gap_func, lmd))
		# our_errors = calc_errors(betas, cands, d_betas, search_space)
		# approx_error = calc_errors(coefs, ys, d_betas, search_space)

		all_times.append(t2 - t1)
		approx_times.append(t3 - t2)

		# all_errors.append(np.mean(our_errors))
		# approx_errors.append(np.mean(approx_error))


	import pickle
	with open("{}_data.pkl".format(args.loss_type), "wb") as f:
		pickle.dump((all_errors, all_times), f)

	# plt.plot(cands, max_asets)
	# plt.legend()
	# plt.xlabel("Dimension")
	# plt.ylabel("Size of Active Set")
	# plt.title(args.loss_type)
	# plt.savefig("code/images/{}_aset.png".format(args.loss_type))
	# plt.clf()

	# plt.plot(range_space, all_errors, label="ours")
	# plt.plot(range_space, approx_errors, label="approx")
	# plt.legend()
	# plt.yscale('log')
	# plt.gca().invert_yaxis()
	# plt.xlabel("Dimension")
	# plt.ylabel("L inf Error")
	# plt.title(args.loss_type)
	# plt.savefig("code/images/{}_err.png".format(args.loss_type))
	# plt.clf()

	plt.plot(range_space, our_gaps, label="ours")
	plt.plot(range_space, approx_gaps, label="approx")
	plt.legend()
	plt.yscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel("Dimension")
	plt.ylabel("Duality Gaps")
	plt.title(args.loss_type)
	plt.savefig("code/images/{}_gap.png".format(args.loss_type))
	plt.clf()

	plt.plot(range_space, all_times, label="our algorithm")
	plt.plot(range_space, approx_times, label="approx")
	plt.legend()
	plt.xlabel("Dimension")
	plt.ylabel("Time (s)")
	plt.title(args.loss_type)
	plt.savefig("code/images/{}_time.png".format(args.loss_type))