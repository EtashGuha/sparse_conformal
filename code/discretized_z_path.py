import numpy as np
from code.utils import is_kink
from code.discretized_lmd_path import WarmDiscretePass
from tqdm import tqdm
def DiscretePass(X, Y, prox_func, dual_gap_func, lmd):
	betas = []
	gaps = []
	kinks = []
	for z in np.linspace(min(Y), max(Y), 100):
		temp_y = np.concatenate((Y, [z]))
		beta, gap = prox_func(X, temp_y, lmd, dual_gap_func, tol=1e-12, max_iter=10000)
		
		if len(betas) > 0 and is_kink(beta, betas[-1]):
			kinks.append(z)

		betas.append(beta)
		gaps.append(gap)

	return betas, gaps, kinks, list(np.linspace(min(Y), max(Y), 1000))

def WarmDiscretePass(X, Y,prox_func, dual_gap_func, lmd):
	betas = []
	gaps = []
	kinks = []
	duals = []
	kink_betas = []
	for z in tqdm(np.linspace(min(Y), max(Y), 100)):
		temp_y = np.concatenate((Y, [z]))
		if len(betas) != 0:
			beta, gap, dual = prox_func(X, temp_y, lmd, dual_gap_func, tol=1e-16, initial_beta = betas[-1], max_iter=10000)
		else:
			beta, gap, dual = prox_func(X, temp_y, lmd, dual_gap_func, tol=1e-16, max_iter=10000)
		
		tmp_aset = np.where(abs(beta) > 1e-8)[0]
		tmp_naset = np.where(abs(beta) < 1e-8)[0]
		beta[tmp_naset] = 0
		if len(betas) > 0 and is_kink(beta, betas[-1]):
			kinks.append(z)
			kink_betas.append(beta)

		betas.append(beta)
		gaps.append(gap)
		duals.append(dual)
	return betas, gaps, kinks, kink_betas, list(np.linspace(min(Y), max(Y), 100)), duals

def WarmDiscretePass(X, Y,prox_func, dual_gap_func, lmd, tol=1e-16):
	betas = []
	gaps = []
	kinks = []
	duals = []
	ranks = []
	kink_betas = []
	for z in tqdm(np.linspace(min(Y), max(Y), 1000)):
		temp_y = np.concatenate((Y, [z]))
		if len(betas) != 0:
			beta, gap, dual = prox_func(X, temp_y, lmd, dual_gap_func, tol=tol, initial_beta = betas[-1], max_iter=10000)
		else:
			beta, gap, dual = prox_func(X, temp_y, lmd, dual_gap_func, tol=tol, max_iter=10000)

		residuals = abs(temp_y - X @ beta)
		rank = len(np.where(residuals > residuals[-1])[0])
		tmp_aset = np.where(abs(beta) > 1e-8)[0]
		tmp_naset = np.where(abs(beta) < 1e-8)[0]
		beta[tmp_naset] = 0
		if len(betas) > 0 and is_kink(beta, betas[-1]):
			kinks.append(z)
			kink_betas.append(beta)

		ranks.append(rank)
		betas.append(beta)
		gaps.append(gap)
		duals.append(dual)

	return betas, gaps, kinks, kink_betas, list(np.linspace(min(Y), max(Y), 100)), duals, ranks

def GridHomotopy(X, Y, alpha, prox_func, dual_gap_func, lmd, tol=1e-16):
	_, _, _, _, points, _, ranks = WarmDiscretePass(X, Y,prox_func, dual_gap_func, lmd, tol=tol)

	bounds = []
	in_interval = False

	num_ele = len(Y)

	for idx in range(len(points)):
		if not in_interval and 1/num_ele * ranks[idx] > alpha:
			bounds.append(points[idx])
			in_interval = True
		if in_interval  and 1/num_ele * ranks[idx] < alpha:
			bounds.append(points[idx])
			in_interval = False

	return min(bounds), max(bounds)

