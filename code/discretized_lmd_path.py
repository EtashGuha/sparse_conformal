import numpy as np
from code.utils import is_kink

def DiscretePass(X, Y, prox_func, dual_gap_func, lmd_max):
	betas = []
	gaps = []
	kinks = []
	for lmd in np.linspace(0, lmd_max, 100):
		beta, gap = prox_func(X, Y, lmd, dual_gap_func,  tol=1e-12, max_iter=100000)
		
		if len(betas) > 0 and is_kink(beta, betas[-1]):
			kinks.append(lmd)

		betas.append(beta)
		gaps.append(gap)
	return betas, gaps, kinks

def WarmDiscretePass(X, Y, prox_func, dual_gap_func, lmd_max):
	betas = []
	gaps = []
	kinks = []
	duals = []
	for lmd in np.linspace(0, lmd_max, 100):
		if len(betas) != 0:
			beta, gap, dual = prox_func(X, Y, lmd, dual_gap_func, tol=1e-12, initial_beta = betas[-1], max_iter=100000)
		else:
			beta, gap, dual = prox_func(X, Y, lmd, dual_gap_func, tol=1e-12, max_iter=100000)

		if len(betas) > 0 and is_kink(beta, betas[-1]):
			kinks.append(lmd)
		betas.append(beta)
		gaps.append(gap)
		duals.append(dual)
		
	return betas, gaps, kinks, list(np.linspace(0, lmd_max, 100)), duals
