from calendar import c
import copy
from re import S
from code.approx_homotopy import calc_full_homo
import numpy as np
import scipy
import argparse 
import random
from sklearn.datasets import make_regression
from scipy.optimize import minimize
from code.FenCon import FenCon
from code.utils import generate_stdp_dataset, is_kink, lasso_loss, lasso_gradient, inv_lasso_gradient, inv_lasso_gradient, logit_grad, inv_logit_grad, assymetric_loss, assymetric_gradient, inv_assymetric_gradient, robust_loss, robust_gradient, inv_robust_gradient, prox_lasso, prox_assymetric, prox_robust, plot_path, plot_single_path, plot_again, lasso_gradient_dbeta, lasso_gradient_dy, assymetric_gradient_dbeta, assymetric_gradient_dy, robust_gradient_dbeta, robust_gradient_dy, plot_really_again, ja_inv_lasso_gradient, ja_inv_assymetric_gradient, ja_inv_robust_gradient
from code.discretized_z_path import WarmDiscretePass
from code.generalized_z_path import ZPath
import time
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets, ensemble
from code.measure_perf import calc_duality_gaps
import multiprocessing as mp
from tqdm import tqdm
import os
import signal
from contextlib import contextmanager
import os
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def solve(X, Y, method, loss_type, lmd, idx):
	search_space = np.linspace(min(Y), max(Y), 25)
	if method == "ours":
		if loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-6, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			cands, betas, duals = assymetric_cp.solve(X, Y[:-1])
			t2 = time.time()
			gap = assymetric_cp.calc_primals(search_space, X, Y)
		elif loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, lmd=lmd,mirror_tol=1e-3, bisection_tol=1e-4, ja_inverse_model=ja_inv_robust_gradient)
			cands, betas, duals = robust_cp.solve(X, Y[:-1])
			t2 = time.time()
			gap = robust_cp.calc_primals(search_space, X, Y)
		elif loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-6, bisection_tol=1e-8, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			cands, betas, duals = lasso_cp.solve(X, Y[:-1])
			t2 = time.time()
			# gap = lasso_cp.calc_primals(search_space, X, Y)
			dual_gap_func = lasso_cp.duality_gap
			primal = lasso_cp.primal
			chosen_proximal = lasso_cp.proximal_func
		cands.reverse()
		betas.reverse()
		gap = calc_duality_gaps(betas,cands, search_space, X, Y, dual_gap_func, chosen_proximal, primal, lmd)
		comp_time = t2 - t1
		num_kinks = len(cands)
	elif method == "approx":
		if args.loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			coefs, ys = calc_full_homo(X, Y, lmd=lmd, dual_gap_func=assymetric_cp.duality_gap, prox_func=prox_assymetric, epsilon = .00001)
			t2 = time.time()
			dual_gap_func = assymetric_cp.duality_gap
			primal = assymetric_cp.primal
			chosen_proximal = assymetric_cp.proximal_func
		elif args.loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, lmd=lmd,mirror_tol=1e-6, bisection_tol=1e-11, ja_inverse_model=ja_inv_robust_gradient)
			coefs, ys = calc_full_homo(X, Y, lmd=lmd, dual_gap_func=robust_cp.duality_gap, prox_func=prox_robust, epsilon =.00001)
			t2 = time.time()
			dual_gap_func = robust_cp.duality_gap
			primal = robust_cp.primal
			chosen_proximal = robust_cp.proximal_func
		elif args.loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			coefs, ys = calc_full_homo(X, Y, lmd=lmd, dual_gap_func=lasso_cp.duality_gap, epsilon =.00001)
			t2 = time.time()
			dual_gap_func = lasso_cp.duality_gap
			primal = lasso_cp.primal
			chosen_proximal = lasso_cp.proximal_func
		coefs.reverse()
		ys.reverse()
		# betas, cands, search_space, X, Y, dual_gap_func, proximal_func, primal, lmd)
		gap = calc_duality_gaps(coefs, ys, search_space, X, Y, dual_gap_func, chosen_proximal, primal, lmd)
		comp_time = t2 - t1
		num_kinks = 0
		for idx in range(len(coefs) - 1):
			if is_kink(coefs[idx], coefs[idx+1]):
				num_kinks += 1

	elif method == "grid":
		if args.loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_assymetric, assymetric_cp.duality_gap, lmd,  tol=1e-9)
			t2 = time.time()
			dual_gap_func = assymetric_cp.duality_gap
		elif args.loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, lmd=lmd,mirror_tol=1e-3, bisection_tol=1e-11, ja_inverse_model=ja_inv_robust_gradient)
			d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_robust, robust_cp.duality_gap, lmd,  tol=1e-4)
			t2 = time.time()
			dual_gap_func = robust_cp.duality_gap
		elif args.loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_lasso, lasso_cp.duality_gap, lmd, tol=1e-10)
			t2 = time.time()
			dual_gap_func = lasso_cp.duality_gap
		coefs.reverse()
		ys.reverse()
		gap = calc_duality_gaps(d_betas, search_space, search_space, X, Y, dual_gap_func, lmd)
		comp_time = t2 - t1
		num_kinks = 0
		for idx in range(len(coefs) - 1):
			if is_kink(coefs[idx], coefs[idx+1]):
				num_kinks += 1
	return gap, comp_time, num_kinks, idx

def random_perm(X, Y):
	swap = random.randint(0, len(X) - 2)
	x_swap = copy.deepcopy(X[swap])
	y_swap = copy.deepcopy(Y[swap])
	X[swap] = X[-1]
	Y[swap] = Y[-1]
	X[-1] = x_swap
	Y[-1] = y_swap
	return X, Y

def solve_dataset(dataset, method, loss_type):
	pool = mp.Pool(10)
	vals = []
	sols = []
	counter = 0
	for X, Y, true_beta, lmd in tqdm(dataset):
		try:
			with time_limit(600):
				vals.append(solve(X, Y, method, loss_type, lmd, counter))
		except TimeoutException as e:
			print("Timed")
			vals.append((1, 300, 0, counter))
		
		# vals.append(pool.apply_async(func=solve, args=(X, Y, method, loss_type, lmd, counter)))
		counter += 1

	pool.close()
	pool.join()

	for val in tqdm(vals):
		gap, comp_time, num_kinks, idx = val
		sols.append((gap, comp_time, num_kinks, idx))
	
	pool.close()
	pool.terminate()
	pool.join()

	sols.sort(key=lambda x: x[-1])

	gaps, comp_times, num_kinks, idxes = list(zip(*sols))

	return gaps, comp_times, num_kinks

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--loss-type", choices=["assymetric", "lasso", "robust"])
	parser.add_argument("--dataset", choices=["synthetic", "diabetes", "cancer", "climate", "cali", "friedman", "friedman2","large"])

	args = parser.parse_args()
	our_results = dict()
	approx_results = dict()
	output_file = "results/homotopy_{}_{}_results.pkl".format(args.dataset, args.loss_type)
	if os.path.exists(output_file):
		with open(output_file, "rb") as f:
			our_results, approx_results = pickle.load(f)
	
	if args.dataset == 'synthetic':
		with open('data/synthetic.pkl', "rb") as f:
			dataset = pickle.load(f)
		# for dim in list(dataset.keys()):
		# 	if dim not in our_results:
		dataset = dataset[100][:5]
		# gaps, comp_times, num_kinks = solve_dataset(dataset, "ours", args.loss_type)
		# our_results = (gaps, comp_times, num_kinks)
		# if dim not in approx_results:

		# breakpoint()
		gaps, comp_times, num_kinks = solve_dataset(dataset, "approx", args.loss_type)
		approx_results = (gaps, comp_times, num_kinks)

		with open(output_file, "wb") as f:
			pickle.dump((our_results, approx_results), f)
	elif args.dataset == "diabetes":
		diabetes = datasets.load_diabetes()
		X, Y = diabetes.data, diabetes.target
		transformed_X = X/X.max(axis=0)
		transformed_Y = (Y - Y[:-1].mean())/Y[:-1].std()
		lmd = np.linalg.norm(transformed_X[:-1].T @ lasso_gradient(np.zeros(len(transformed_X[:-1])), transformed_Y[:-1]), ord=np.inf)/20
		dataset = []
		for _ in range(1):
			new_X, new_Y = random_perm(copy.deepcopy(transformed_X), copy.deepcopy(transformed_Y))
			dataset.append((new_X, new_Y, None, lmd))
		gaps, comp_times, num_kinks = solve_dataset(dataset, "ours", args.loss_type)
		our_results["ours"] = (gaps, comp_times, num_kinks)
		gaps, comp_times, num_kinks = solve_dataset(dataset, "approx", args.loss_type)
		approx_results["approx"] = (gaps, comp_times, num_kinks)
		breakpoint()
	elif args.dataset == "cali":
		cali = datasets.fetch_california_housing()
		X, Y = cali.data, cali.target
		transformed_X = X/X.max(axis=0)
		transformed_Y = (Y - Y[:-1].mean())/Y[:-1].std()
		lmd = np.linalg.norm(transformed_X[:-1].T @ lasso_gradient(np.zeros(len(transformed_X[:-1])), transformed_Y[:-1]), ord=np.inf)/20
		dataset = []
		for _ in range(30):
			new_X, new_Y = random_perm(copy.deepcopy(transformed_X), copy.deepcopy(transformed_Y))
			dataset.append((new_X, new_Y, None, lmd))
		gaps, comp_times, num_kinks = solve_dataset(dataset, "ours", args.loss_type)
		our_results["ours"] = (gaps, comp_times, num_kinks)
		gaps, comp_times, num_kinks = solve_dataset(dataset, "approx", args.loss_type)
		approx_results["approx"] = (gaps, comp_times, num_kinks)
	elif args.dataset == "cali":
		cali = datasets.fetch_california_housing()
		X, Y = cali.data, cali.target
		transformed_X = X/X.max(axis=0)
		transformed_Y = (Y - Y[:-1].mean())/Y[:-1].std()
		lmd = np.linalg.norm(transformed_X[:-1].T @ lasso_gradient(np.zeros(len(transformed_X[:-1])), transformed_Y[:-1]), ord=np.inf)/20
		dataset = []
		for _ in range(10):
			new_X, new_Y = random_perm(copy.deepcopy(transformed_X), copy.deepcopy(transformed_Y))
			dataset.append((new_X, new_Y, None, lmd))
		gaps, comp_times, num_kinks = solve_dataset(dataset, "ours", args.loss_type)
		our_results["ours"] = (gaps, comp_times, num_kinks)
		gaps, comp_times, num_kinks = solve_dataset(dataset, "approx", args.loss_type)
		approx_results["approx"] = (gaps, comp_times, num_kinks)
	elif args.dataset == "friedman":
		friedman = datasets.make_friedman1()
		X, Y = friedman[0], friedman[1]
		transformed_X = X/X.max(axis=0)
		transformed_Y = (Y - Y[:-1].mean())/Y[:-1].std()
		lmd = np.linalg.norm(transformed_X[:-1].T @ lasso_gradient(np.zeros(len(transformed_X[:-1])), transformed_Y[:-1]), ord=np.inf)/20
		dataset = []
		for _ in range(1):
			new_X, new_Y = random_perm(copy.deepcopy(transformed_X), copy.deepcopy(transformed_Y))
			dataset.append((new_X, new_Y, None, lmd))
		gaps, comp_times, num_kinks = solve_dataset(dataset, "ours", args.loss_type)
		our_results["ours"] = (gaps, comp_times, num_kinks)
		gaps, comp_times, num_kinks = solve_dataset(dataset, "approx", args.loss_type)
		approx_results["approx"] = (gaps, comp_times, num_kinks)
	elif args.dataset == "friedman2":
		friedman = datasets.make_friedman2()
		X, Y = friedman[0], friedman[1]
		transformed_X = X/X.max(axis=0)
		transformed_Y = (Y - Y[:-1].mean())/Y[:-1].std()
		lmd = np.linalg.norm(transformed_X[:-1].T @ lasso_gradient(np.zeros(len(transformed_X[:-1])), transformed_Y[:-1]), ord=np.inf)/20
		dataset = []
		for _ in range(1):
			new_X, new_Y = random_perm(copy.deepcopy(transformed_X), copy.deepcopy(transformed_Y))
			dataset.append((new_X, new_Y, None, lmd))
		gaps, comp_times, num_kinks = solve_dataset(dataset, "ours", args.loss_type)
		our_results["ours"] = (gaps, comp_times, num_kinks)
		gaps, comp_times, num_kinks = solve_dataset(dataset, "approx", args.loss_type)
		approx_results["approx"] = (gaps, comp_times, num_kinks)
	elif args.dataset == "large":
		X, Y, beta_true = generate_stdp_dataset(100, 20, -20, 20)
		lmd = np.linalg.norm(X[:-1].T @ lasso_gradient(np.zeros(len(X[:-1])), Y[:-1]), ord=np.inf)/20
		dataset = [(X, Y, None, lmd)]
		gaps, comp_times, num_kinks = solve_dataset(dataset, "ours", args.loss_type)
		our_results["ours"] = (gaps, comp_times, num_kinks)
		gaps, comp_times, num_kinks = solve_dataset(dataset, "approx", args.loss_type)
		approx_results["approx"] = (gaps, comp_times, num_kinks)
		breakpoint()
	
	
	
	
	
	
	
	
	with open(output_file, "wb") as f:
		pickle.dump((our_results, approx_results), f)




	breakpoint()