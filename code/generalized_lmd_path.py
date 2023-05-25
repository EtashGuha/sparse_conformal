import argparse
from tkinter import N
import numpy as np
import copy
import time
from utils import generate_stdp_dataset, lasso_loss, lasso_gradient, inv_lasso_gradient, inv_lasso_gradient, logit_grad, inv_logit_grad, assymetric_loss, assymetric_gradient, inv_assymetric_gradient, robust_loss, robust_gradient, inv_robust_gradient, prox_lasso, prox_assymetric, prox_robust, plot_path
from FenCon import FenCon
from discretized_lmd_path import WarmDiscretePass


class LmdPath(FenCon):
	def __init__(self, loss, model, inverse_model, proximal_func):
		super().__init__(loss, model, inverse_model)
		self.lmds = []
		self.proximal_func = proximal_func
		

	def solve(self, X, Y):
		self.initialize(X, Y)
		self.thetas = []

		self.lmds = []
		self.betas = []
		self.duals = []
		
		while not np.isclose(self.lmd, 0.) and len(self.naset) != 0:
			status = self.update(X, Y)
			if status == -1:
				break
		
		self.update_beta_lambda(X, Y)
		return self.lmds, self.betas, self.duals
		
	def update_beta_lambda(self, X, Y):

		self.J_A = np.linalg.solve(X[:, self.aset].T @ X[:, self.aset], X[:, self.aset].T)
		t1 = time.time()

		self.beta = self.get_mirror(X, Y, self.proximal_func)
		# self.beta[self.naset] *= 0
		t2 = time.time()
		
		Xbeta = X @ self.beta
		grad_f = self.model(Xbeta, Y)
		self.theta_estim = - grad_f / max(np.linalg.norm(X.T.dot(grad_f), ord=np.inf), self.lmd)

		# prox_beta , prox_gap = self.get_prox(X, Y)
		t3 = time.time()
		# ada_beta, ada_gap = self.get_ada(X, Y)
		t4 = time.time()
		self.total_ada += t4-t3
		self.total_prox += t3 - t2
		self.total_mirror += t2 - t1
		
		# print("Ada Gap: {} Ada Time: {}".format(ada_gap, t4-t3))
		# print("Prox Gap: {} Prox Time: {}".format(prox_gap, t3-t2))
		# self.ada_gaps.append(ada_gap)
		# self.prox_gaps.append(prox_gap)
		# self.beta[self.naset] *= 0

		Xbetamap = X @ self.beta
		grad_fmap = self.model(Xbetamap, Y)
		self.theta_estim = - grad_fmap / max(np.linalg.norm(X.T.dot(grad_fmap), ord=np.inf), self.lmd)

		self.lmds.append(self.lmd)
		self.betas.append(copy.deepcopy(self.beta))
		self.duals.append(copy.deepcopy(self.theta_estim))
		duality_gap = self.duality_gap(X, Y)
		self.mirror_gaps.append(duality_gap)
		print("mirror Gap: {} mirror time: {}".format(duality_gap, t2-t1))
		print("dual gap: {}".format(duality_gap))
		self.eta = np.sign(-X.T @ self.model(X @ self.beta, Y))

	def update(self, X, Y):
		self.update_beta_lambda(X, Y)

		#Calc nonzero to zero lambda
		exits = []
		for j in self.aset:
			idx_j = self.aset.index(j)
			def calc_next_lam_nz(x):
				return (self.J_A @ self.inverse_model(-x * self.theta_estim, Y))[idx_j]
			exits.append(self.do_bisection(calc_next_lam_nz, 0, self.lmd))
			
		best_nz_to_z_index = self.aset[np.argmax(exits)] if len(exits) > 0 else None
		nz_to_z_lam_next = max(exits) if len(exits) > 0 else -1 * float('inf')
		
		entrances_neg = []
		entrances_pos = []
		#Calc zero to nnonzero lambda
		
		XJ_A = X[:, self.aset] @ self.J_A
		for feat_idx in self.naset:
			
			def temp_calc_next_pos(x):
				val = -X[:, feat_idx].T @ (self.model(XJ_A @ self.inverse_model(-x * self.theta_estim, Y), Y)) - x
				return val

			def temp_calc_next_neg(x):
				val = -X[:, feat_idx].T @ (self.model(XJ_A @ self.inverse_model(-x * self.theta_estim, Y), Y)) + x
				return val

			entrances_pos.append(self.do_bisection(temp_calc_next_pos, 1e-6, self.lmd))
			entrances_neg.append(self.do_bisection(temp_calc_next_neg, 1e-6, self.lmd))

		entrances = [max(entrances_neg[i], entrances_pos[i]) for i in range(len(self.naset))]
		best_z_to_nz_index = self.naset[np.argmax(entrances)]
		z_to_nz_lam_next = max(entrances)

		if z_to_nz_lam_next < 0  and nz_to_z_lam_next < 0:
			return -1

		if z_to_nz_lam_next > nz_to_z_lam_next:
			self.aset.append(best_z_to_nz_index)
			self.naset.remove(best_z_to_nz_index)
			self.lmd = z_to_nz_lam_next
		else:
			self.aset.remove(best_nz_to_z_index)
			self.naset.append(best_nz_to_z_index)
			self.lmd = nz_to_z_lam_next
		return 1

	def initialize(self, X, Y):
		self.dy_dz = np.zeros_like(Y)
		self.dy_dz[-1] = 1
		self.n_samples, self.n_features = X.shape
		self.beta = np.zeros(self.n_features)
		grad_f = self.model(np.zeros(self.n_samples), Y)
		XTgrad_f = X.T @ grad_f
		all_lams = XTgrad_f
		self.aset = [np.argmax(abs(all_lams))]
		self.naset = list(set(range(self.n_features)) - set(self.aset))
		self.lmd = abs(all_lams[self.aset][0])
		self.eta = np.sign(-XTgrad_f)
		self.J_A = np.linalg.solve(X[:, self.aset].T @ X[:, self.aset], X[:, self.aset].T)

		duality_gap = self.duality_gap(X, Y)
		print("dual gap: {}".format(duality_gap))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dim", type=int, default=10)
	parser.add_argument("--num-examples", type=int, default=5)
	parser.add_argument("--max-val", type=int, default=1)
	parser.add_argument("--min-val", type=int, default=-1)
	parser.add_argument("--loss-type", choices=["assymetric", "lasso", "robust"])

	args = parser.parse_args()
	X, Y, beta_true = generate_stdp_dataset(args.dim, args.num_examples, args.min_val, args.max_val)

	

	if args.loss_type == "assymetric":
		assymetric_cp = LmdPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric)
		lams, betas, duals = assymetric_cp.solve(X, Y)
		dbetas, dgaps, dkinks, search_space, d_duals = WarmDiscretePass(X, Y, prox_assymetric, assymetric_cp.duality_gap, lams[0])
	elif args.loss_type == "robust":
		robust_cp = LmdPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust)
		lams, betas, duals = robust_cp.solve(X, Y)
		dbetas, dgaps, dkinks, search_space, d_duals = WarmDiscretePass(X, Y, prox_robust, robust_cp.duality_gap, lams[0])
	elif args.loss_type == "lasso":
		lasso_cp = LmdPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso)
		lams, betas, duals = lasso_cp.solve(X, Y)
		dbetas, dgaps, dkinks, search_space, d_duals = WarmDiscretePass(X, Y, prox_lasso, lasso_cp.duality_gap, lams[0])

	plot_path(dbetas, search_space, betas, lams, 'lambdas', name="{}_{}".format(args.loss_type, 'lmd'))
	plot_path(d_duals, search_space, duals, lams, 'candidates', name="{}_{}".format(args.loss_type, 'lmd_duals'))
	breakpoint()



