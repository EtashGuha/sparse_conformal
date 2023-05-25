from calendar import c
import copy
from re import S
from code.approx_homotopy import calc_full_homo, calc_homotopy_cp
import numpy as np
import scipy
import argparse 
from sklearn.datasets import make_regression
from scipy.optimize import minimize
from code.FenCon import FenCon
from code.utils import generate_stdp_dataset, is_kink, lasso_loss, lasso_gradient, inv_lasso_gradient, inv_lasso_gradient, logit_grad, inv_logit_grad, assymetric_loss, assymetric_gradient, inv_assymetric_gradient, robust_loss, robust_gradient, inv_robust_gradient, prox_lasso, prox_assymetric, prox_robust, plot_path, plot_single_path, plot_again, lasso_gradient_dbeta, lasso_gradient_dy, assymetric_gradient_dbeta, assymetric_gradient_dy, robust_gradient_dbeta, robust_gradient_dy, plot_really_again, ja_inv_lasso_gradient, ja_inv_assymetric_gradient, ja_inv_robust_gradient
from code.discretized_z_path import WarmDiscretePass, GridHomotopy
from code.generalized_z_path import ZPath
import time
import matplotlib.pyplot as plt
import pickle
import time
from sklearn import datasets, ensemble
from code.measure_perf import calc_duality_gaps
from rootCP.rootcp import rootCP, models
import multiprocessing as mp
from tqdm import tqdm
from rootCP.rootcp import rootCP, models, tools
import signal
from contextlib import contextmanager
import os
class TimeoutException(Exception): pass
import random

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

def solve(X, Y, method, loss_type, lmd, idx, alpha):
	search_space = np.linspace(min(Y), max(Y), 100)
	if method == "ours":
		if loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-6, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			bottom, top = assymetric_cp.get_conf_interval(X, Y[:-1], alpha)
			t2 = time.time()
		elif loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, lmd=lmd,mirror_tol=1e-4, bisection_tol=1e-4, ja_inverse_model=ja_inv_robust_gradient)
			bottom, top = robust_cp.get_conf_interval(X, Y[:-1], alpha)
			t2 = time.time()
		elif loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-7, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			bottom, top = lasso_cp.get_conf_interval(X, Y[:-1], alpha)
			t2 = time.time()
		if Y[len(X) - 1] < top and Y[len(X) - 1] > bottom:
			covered = 1
		else:
			covered = 0
		
		length = top - bottom
		time_taken = t2 - t1
	elif method == "approx":
		if loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			bottom, top = calc_homotopy_cp(X, Y, alpha, lmd, dual_gap_func=assymetric_cp.duality_gap, prox_func=prox_assymetric, epsilon = .1)
			t2 = time.time()
		elif loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, lmd=lmd,mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_robust_gradient)
			bottom, top = calc_homotopy_cp(X, Y, alpha, lmd, dual_gap_func=robust_cp.duality_gap, prox_func=prox_robust, epsilon =.1)
			t2 = time.time()
		elif loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			bottom, top = calc_homotopy_cp(X, Y, alpha, lmd, dual_gap_func=lasso_cp.duality_gap, epsilon =1e-1)
			t2 = time.time()
		if Y[len(X) - 1] < top and Y[len(X) - 1] > bottom:
			covered = 1
		else:
			covered = 0
		
		length = top - bottom
		time_taken = t2 - t1
	elif method == "root":

		if loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_assymetric, lmd, assymetric_cp.duality_gap, 1e-9, 10000)
			cp = rootCP.conformalset(X, Y[:-1], regression_model, alpha=alpha)
			bottom = cp[0]
			top = cp[1]
			t2 = time.time()
		elif loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_robust_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_robust, lmd, robust_cp.duality_gap, 1e-10, 10000)
			cp = rootCP.conformalset(X, Y[:-1], regression_model, alpha=alpha)
			bottom = cp[0]
			top = cp[1]
			t2 = time.time()
		elif loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_lasso, lmd, lasso_cp.duality_gap, 1e-10, 10000)
			cp = rootCP.conformalset(X, Y[:-1], regression_model, alpha=alpha)
			bottom = cp[0]
			top = cp[1]
			t2 = time.time()
		assert(bottom < top)
		if Y[len(X) - 1] < top and Y[len(X) - 1] > bottom:
			covered = 1
		else:
			covered = 0
		
		length = top - bottom
		time_taken = t2 - t1
	
	elif method == "split":

		if loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_assymetric, lmd, assymetric_cp.duality_gap, 1e-9, 10000)
			try:
				cp = tools.splitCP(X, Y[:-1], regression_model, alpha=alpha)
				bottom = cp[0]
				top = cp[1]
			except:
				bottom = min(Y)
				top = max(Y)
			t2 = time.time()
		elif loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_robust_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_robust, lmd, robust_cp.duality_gap, 1e-8, 10000)
			try:
				cp = tools.splitCP(X, Y[:-1], regression_model, alpha=alpha)
				bottom = cp[0]
				top = cp[1]
			except:
				bottom = min(Y)
				top = max(Y)
			t2 = time.time()
		elif loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_lasso, lmd, lasso_cp.duality_gap, 1e-10, 10000)
			try:
				cp = tools.splitCP(X, Y[:-1], regression_model, alpha=alpha)
				bottom = cp[0]
				top = cp[1]
			except:
				bottom = min(Y)
				top = max(Y)
			t2 = time.time()
		if Y[len(X) - 1] < top and Y[len(X) - 1] > bottom:
			covered = 1
		else:
			covered = 0
		
		length = top - bottom
		time_taken = t2 - t1
	
	elif method == "oracle":

		if loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_assymetric, lmd, assymetric_cp.duality_gap, 1e-9, 10000)
			cp = tools.oracleCP(X, Y, regression_model, alpha=alpha)
			bottom = cp[0]
			top = cp[1]
			t2 = time.time()
		elif loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_robust_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_robust, lmd, robust_cp.duality_gap, 1e-9, 10000)
			cp = tools.oracleCP(X, Y, regression_model, alpha=alpha)
			bottom = cp[0]
			top = cp[1]
			t2 = time.time()
		elif loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			regression_model = models.proximal_regressor(prox_lasso, lmd, lasso_cp.duality_gap, 1e-10, 10000)
			cp = tools.oracleCP(X, Y, regression_model, alpha=alpha)
			bottom = cp[0]
			top = cp[1]
			t2 = time.time()
		if Y[len(X) - 1] < top and Y[len(X) - 1] > bottom:
			covered = 1
		else:
			covered = 0
		
		length = top - bottom
		time_taken = t2 - t1

	elif method == "grid":
		if loss_type == "assymetric":
			t1 = time.time()
			assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_assymetric_gradient, lmd=lmd)
			bottom, top = GridHomotopy(X, Y[:-1], alpha, prox_assymetric, assymetric_cp.duality_gap, lmd, tol=1e-8)
			t2 = time.time()
		elif loss_type == "robust":
			t1 = time.time()
			robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, mirror_tol=1e-9, bisection_tol=1e-10, ja_inverse_model=ja_inv_robust_gradient, lmd=lmd)
			bottom, top = GridHomotopy(X, Y[:-1], alpha, prox_robust, robust_cp.duality_gap, lmd, tol=1e-8)
			t2 = time.time()
		elif loss_type == "lasso":
			t1 = time.time()
			lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-10, ja_inverse_model=ja_inv_lasso_gradient, lmd=lmd)
			bottom, top = GridHomotopy(X, Y[:-1], alpha, prox_lasso, lasso_cp.duality_gap, lmd, tol=1e-8)
			t2 = time.time()
		if Y[len(X) - 1] < top and Y[len(X) - 1] > bottom:
			covered = 1
		else:
			covered = 0
		length = top - bottom
		time_taken = t2 - t1
	return covered, length, time_taken, idx

def solve_dataset(dataset, method, loss_type, alpha):
	pool = mp.Pool(10)
	vals = []
	sols = []
	counter = 0
	for X, Y, true_beta, lmd in dataset:
		try:
			with time_limit(500):
				# try:
				vals.append(solve(X, Y, method, loss_type, lmd, counter, alpha))
				# except:
				# 	vals.append((0, max(Y) - min(Y), 500, counter))
		except TimeoutException as e:
			vals.append((0, max(Y) - min(Y), 500, counter))

		
		# vals.append(pool.apply_async(func=solve, args=(X, Y, method, loss_type, lmd, counter)))
		counter += 1

	pool.close()
	pool.join()

	for val in tqdm(vals):
		covered, length, time_taken, idx = val
		sols.append((covered, length, time_taken, idx))
	
	pool.close()
	pool.terminate()
	pool.join()

	sols.sort(key=lambda x: x[-1])
	covereds, lengths, times, idxes = list(zip(*sols))

	return covereds, lengths, times

def random_perm(X, Y):
	swap = random.randint(0, len(X) - 2)
	x_swap = copy.deepcopy(X[swap])
	y_swap = copy.deepcopy(Y[swap])
	X[swap] = X[-1]
	Y[swap] = Y[-1]
	X[-1] = x_swap
	Y[-1] = y_swap
	return X, Y

if __name__ == "__main__":
	def restricted_float(x):
		try:
			x = float(x)
		except ValueError:
			raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

		if x < 0.0 or x > 1.0:
			raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
		return x
	parser = argparse.ArgumentParser()
	parser.add_argument("--loss-type", choices=["assymetric", "lasso", "robust"])
	parser.add_argument("--dataset", choices=["synthetic", "syntheticsmall", "diabetes", "cancer", "climate", "cali", "friedman2"])
	parser.add_argument("--alpha", type=restricted_float)

	args = parser.parse_args()
	our_results = dict()
	approx_results = dict()
	root_results = dict()
	split_results = dict()
	grid_results = dict()
	oracle_results = dict()
	output_file_name = "results/cp_{}_{}_{}a_results.pkl".format(args.dataset, args.loss_type, args.alpha)
	if os.path.exists(output_file_name):
		with open(output_file_name, "rb") as f:
			our_results, approx_results, root_results, split_results, grid_results, oracle_results = pickle.load(f)
		
	def save_model():
		with open(output_file_name, "wb") as f:
			pickle.dump((our_results, approx_results, root_results, split_results, grid_results, oracle_results), f)


	if args.dataset == 'syntheticsmall':
		with open('data/synthetic_small.pkl', "rb") as f:
			dataset = pickle.load(f)
		for dim in dataset.keys():

			# if dim not in our_results:
			# 	covereds, lengths, times = solve_dataset(dataset[dim], "ours", args.loss_type, args.alpha)
			# 	our_results[dim] = (covereds, lengths, times)
			# 	save_model()
			
			# if dim not in approx_results:
			# 	covereds, lengths, times = solve_dataset(dataset[dim], "approx", args.loss_type, args.alpha)
			# 	approx_results[dim] = (covereds, lengths, times)
			# 	save_model()

			# if dim not in root_results:
			# 	covereds, lengths, times = solve_dataset(dataset[dim], "root", args.loss_type, args.alpha)
			# 	root_results[dim] = (covereds, lengths, times)
			# 	save_model()

			# if dim not in split_results:
			# 	covereds, lengths, times = solve_dataset(dataset[dim], "split", args.loss_type, args.alpha)
			# 	split_results[dim] = (covereds, lengths, times)
			# 	save_model()

			covereds, lengths, times = solve_dataset(dataset[dim], "grid", args.loss_type, args.alpha)
			grid_results[dim] = (covereds, lengths, times)
			save_model()
			breakpoint()
			if dim not in oracle_results:
				covereds, lengths, times = solve_dataset(dataset[dim], "oracle", args.loss_type, args.alpha)
				oracle_results[dim] = (covereds, lengths, times)
				save_model()


	elif args.dataset == "diabetes":
		diabetes = datasets.load_diabetes()
		X, Y = diabetes.data, diabetes.target
		transformed_X = X/np.max(X)
		transformed_Y = (Y - Y[:-1].mean())/Y[:-1].std()
		lmd = np.linalg.norm(transformed_X[:-1].T @ lasso_gradient(np.zeros(len(transformed_X[:-1])), transformed_Y[:-1]), ord=np.inf)/20
		dataset = []
		for _ in range(15):
			new_X, new_Y = random_perm(copy.deepcopy(transformed_X), copy.deepcopy(transformed_Y))
			dataset.append((new_X, new_Y, None, lmd))
		# covereds, lengths, times = solve_dataset(dataset, "ours", args.loss_type, args.alpha)
		# our_results["ours"] = (covereds, lengths, times)
		# covereds, lengths, times = solve_dataset(dataset, "approx", args.loss_type, args.alpha)
		# approx_results["approx"] = (covereds, lengths, times)
		# covereds, lengths, times = solve_dataset(dataset, "root", args.loss_type, args.alpha)
		# root_results["root"] = (covereds, lengths, times)
		# covereds, lengths, times = solve_dataset(dataset, "split", args.loss_type, args.alpha)
		# split_results["split"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "grid", args.loss_type, args.alpha)
		grid_results["grid"] = (covereds, lengths, times)
		breakpoint()
		covereds, lengths, times = solve_dataset(dataset, "oracle", args.loss_type, args.alpha)
		oracle_results["oracle"] = (covereds, lengths, times)
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
		covereds, lengths, times = solve_dataset(dataset, "ours", args.loss_type, args.alpha)
		our_results["ours"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "approx", args.loss_type, args.alpha)
		approx_results["approx"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "root", args.loss_type, args.alpha)
		root_results["root"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "split", args.loss_type, args.alpha)
		split_results["split"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "grid", args.loss_type, args.alpha)
		grid_results["grid"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "oracle", args.loss_type, args.alpha)
		oracle_results["oracle"] = (covereds, lengths, times)
	elif args.dataset == "friedman2":
		friedman = datasets.make_friedman2()
		X, Y = friedman[0], friedman[1]
		transformed_X = X/X.max(axis=0)
		transformed_Y = (Y - Y[:-1].mean())/Y[:-1].std()
		lmd = np.linalg.norm(transformed_X[:-1].T @ lasso_gradient(np.zeros(len(transformed_X[:-1])), transformed_Y[:-1]), ord=np.inf)/20
		dataset = []
		for _ in range(30):
			new_X, new_Y = random_perm(copy.deepcopy(transformed_X), copy.deepcopy(transformed_Y))
			dataset.append((new_X, new_Y, None, lmd))
		covereds, lengths, times = solve_dataset(dataset, "ours", args.loss_type, args.alpha)
		our_results["ours"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "approx", args.loss_type, args.alpha)
		approx_results["approx"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "root", args.loss_type, args.alpha)
		root_results["root"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "split", args.loss_type, args.alpha)
		split_results["split"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "grid", args.loss_type, args.alpha)
		grid_results["grid"] = (covereds, lengths, times)
		covereds, lengths, times = solve_dataset(dataset, "oracle", args.loss_type, args.alpha)
		oracle_results["oracle"] = (covereds, lengths, times)

	
	
	
	
	
	
	
	with open("results/cp_{}_{}_{}a_results.pkl".format(args.dataset, args.loss_type, args.alpha), "wb") as f:
		pickle.dump((our_results, approx_results, root_results, split_results, grid_results, oracle_results), f)

	


