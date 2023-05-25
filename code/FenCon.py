from tkinter import N
from turtle import begin_fill
import numpy as np
import scipy
from scipy.optimize import minimize
import copy

class FenCon():
	def __init__(self, loss, model, inverse_model, second_beta, second_y):
		self.model = model 
		self.inverse_model = inverse_model
		self.second_beta = second_beta
		self.second_y = second_y
		self.loss = loss
		self.total_prox = 0
		self.total_ada = 0
		self.total_mirror = 0
		self.prox_gaps = []
		self.ada_gaps = []
		self.mirror_gaps = []
 
	def get_mirror(self, X, Y, proximal_func=None, mirror_tol=1e-12):
		
		if proximal_func is None:
			Xbeta = X @ self.beta
			grad_f = self.model(Xbeta, Y)
			warm_start = - grad_f / max(np.linalg.norm(X.T.dot(grad_f), ord=np.inf), self.lmd)

			def dual_obj(theta):
				reg = np.linalg.norm(X[:, self.aset].T.dot(theta) + self.lmd * self.eta[self.aset])
				return -self.dual(theta, Y) +  reg

			def primal_obj(beta):
				return self.primal(X[:, self.aset], Y, beta)

			res = minimize(dual_obj, warm_start, method="Nelder-Mead", options={"maxiter":100000}, tol=1e-12)
			theta_estim = res.x
			theta_estim = theta_estim / max(np.linalg.norm(X.T.dot(theta_estim), ord=np.inf), self.lmd)
			self.thetas.append(theta_estim)
			warm_start_beta = self.J_A @ self.inverse_model(-self.lmd * theta_estim, Y) 
			augmented_beta = np.zeros_like(self.beta)

		
			# res_primal = minimize(lambda tbeta: self.primal(X, Y, tbeta), augmented_beta, method="Nelder-Mead", options={"maxiter":100000}, tol=1e-12)
			try:
				res_primal = minimize(primal_obj,warm_start_beta, method="Nelder-Mead", options={"maxiter":100000}, tol=1e-10)
			except:
				breakpoint()
			augmented_beta = np.zeros_like(self.beta)
			augmented_beta[self.aset] = res_primal.x

			return augmented_beta
		elif len(self.betas) > 0 and self.prev_aset is not -1:
			augmented_beta = np.zeros_like(self.beta)
			# augmented_beta[self.aset] = warm_start_beta
			dy_dz = np.zeros_like(Y)
			dy_dz[-1] = 1
			#play
			play_tmp_Y = copy.deepcopy(Y)
			play_tmp_Y[-1] = (self.cands[-1] + self.cand)/2

			second_derivative_val = self.second_beta(X[:, self.prev_aset], self.betas[-1][self.prev_aset], play_tmp_Y)
		
			play_db_dz_again = np.linalg.solve(-1 * second_derivative_val, self.second_y(X[:, self.prev_aset], self.betas[-1][self.prev_aset], play_tmp_Y) @ dy_dz) 
			warm_start_beta = play_db_dz_again * (self.cand - self.cands[-1]) + self.betas[-1][self.prev_aset]
			augmented_beta[self.prev_aset] = warm_start_beta
			# print(len(self.aset))
			warm_gap = self.duality_gap(X, Y, beta=augmented_beta)
			trivial_gap = self.duality_gap(X, Y, beta=self.betas[-1])
			# print(min(warm_gap, trivial_gap))
			if warm_gap < trivial_gap:
				aset_beta, gap, dual = proximal_func(X[:, self.aset], Y, self.lmd, self.duality_gap, initial_beta=augmented_beta[self.aset], tol=mirror_tol, max_iter=10000)
				full_warm_beta = np.zeros_like(self.beta)
				full_warm_beta[self.aset] = aset_beta
				beta, gap, dual = proximal_func(X, Y, self.lmd, self.duality_gap, initial_beta=full_warm_beta, tol=mirror_tol, max_iter=10000)
				tmp_naset = np.where(abs(beta) < 1e-12)[0]
				beta[tmp_naset] = 0
			else:
				beta, gap, dual = proximal_func(X, Y, self.lmd, self.duality_gap, initial_beta=self.betas[-1], tol=mirror_tol, max_iter=10000)
				tmp_naset = np.where(abs(beta) < 1e-12)[0]
				beta[tmp_naset] = 0

			if beta is None:
				breakpoint()
			return beta
		else:
			beta, gap, dual = proximal_func(X, Y, self.lmd, self.duality_gap, tol=mirror_tol, max_iter=10000)
			return beta
	def do_bisection(self, funct, left, right, tol=1e-12):
		final_val = -float('inf')
	
		if not np.sign(funct(left)) == np.sign(funct(right)):
			final_val = scipy.optimize.bisect(funct, right, left, rtol=tol, xtol=tol)
	
		# if np.isclose(final_val, right):
		# 	final_val = -float('inf')
		return final_val

	def do_bisection_checker(self, funct, left, right, step_size, dim):
		features = np.asarray(range(dim))

		right_vals = funct(right, features)
		starting_sign = np.sign(funct(right, features))
		is_not_close_to_0 = np.logical_not(np.isclose(right_vals, np.zeros(dim)))
		counter = 1
		prev_left_hat = right
		left_hat = right - counter * step_size

		# breakpoint()
		viable = []
		while left_hat > left:
			left_hat_sign = np.sign(funct(left_hat, features))
			viable = np.where(np.logical_and(starting_sign[features] != left_hat_sign, is_not_close_to_0[features]))[0]
			if len(viable) > 0:
				break
			prev_left_hat = left_hat
			counter *= 2
			left_hat -= counter * step_size

		if len(viable) == 0:
			left_hat_sign = np.sign(funct(left_hat, features))
			viable = np.where(np.logical_and(starting_sign != left_hat_sign, is_not_close_to_0))[0]
		return viable, left_hat, prev_left_hat


	def primal(self, X, Y, beta):
		reg = np.linalg.norm(beta, ord=1)
		loss = self.loss(X.dot( beta), Y)
		return np.sum(loss) + self.lmd * reg

	def dual(self, temptheta, Y):
		theta = -self.lmd * temptheta
		optval = self.inverse_model(theta, Y)
		inner = theta * optval
		fval = self.loss(optval, Y)
		return -np.sum(inner - fval)

	def dual_vec(self, X, Y, beta=None):
		if beta is None:
			Xbeta = X.dot(self.beta)
		else:
			Xbeta = X.dot(beta)
		grad_f = self.model(Xbeta, Y)
		dual_scaling = max(np.linalg.norm(X.T.dot(grad_f), ord=np.inf), self.lmd)

		theta = -grad_f / dual_scaling
		return theta

	def duality_gap(self, X, Y, lmd=None, beta=None, tol=1e-12):
		prev_lmd = self.lmd

		if lmd is not None:
			self.lmd = lmd
		theta = self.dual_vec(X, Y, beta=beta)
		if beta is not None:
			p_obj = self.primal(X, Y, beta)
		else:
			p_obj = self.primal(X, Y, self.beta)
		d_obj = self.dual(theta, Y)

		dual_gap = p_obj - d_obj
		# if dual_gap > tol:
		# 	print('non convergence')

		if lmd is not None:
			self.lmd = prev_lmd
		return dual_gap
