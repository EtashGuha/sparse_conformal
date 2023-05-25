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

class ZPath(FenCon):
	def __init__(self, loss, model, inverse_model, proximal_func, d_second_beta, d_second_y, mirror_tol, ja_inverse_model, bisection_tol=1e-12, lmd=.5, is_lasso=False):
		super().__init__(loss, model, inverse_model, d_second_beta, d_second_y)
		self.lmd = lmd
		self.proximal_func = proximal_func
		self.is_lasso = is_lasso
		self.small_delta = 0
		self.multiplier = 1
		self.bisection_tol = bisection_tol
		self.mirror_tol = mirror_tol
		self.approx_smoothness = 1/2
		self.approx_tolerance = self.bisection_tol
		self.approx_not_used = True
		self.ja_inverse_model = ja_inverse_model
		self.max_aset = []
		self.step_size = 1e-2
		self.prev_aset = -1
		self.cp_tol = .001

	def calc_duality_gaps(self, search_space, X, Y, true_betas):
		print('CALCING')
		betas = self.betas.copy()
		betas.reverse()
		cands = self.cands.copy()
		cands.reverse()
		thetas = self.thetas.copy()
		thetas.reverse()
		duals = self.duals.copy()
		duals.reverse()
		J_As = self.J_As.copy()
		J_As.reverse()
		asets = self.asets.copy()
		asets.reverse()
		Y_z = copy.deepcopy(Y)
		errors = []
		true_gaps = []
		best_gaps = []
		true_primals=[]
		for idx in range(len(search_space)):
			
			real_cand = search_space[idx]
			Y_z[-1] = real_cand
			if real_cand < cands[0]:
				# slopes = (betas[1] - betas[0])/(cands[1] - cands[0])
				# final_beta = betas[0] + slopes * (real_cand - cands[0])
				final_beta = betas[0]
			elif real_cand > cands[-1]:
				# slopes = (betas[-1] - betas[-2])/(cands[-1] - cands[-2])
				# final_beta = betas[-1] + slopes * (real_cand - cands[-1]) 
				final_beta = betas[-1]
			else:
				
				for i in range(len(cands) - 1):
					tmp_y = copy.deepcopy(Y_z)
					tmp_y[-1] = cands[i+1]
					play_db_dz_again = -1 * np.linalg.inv(self.second_beta(X[:, asets[i + 1]], betas[i + 1][asets[i + 1]], tmp_y)) @ self.second_y(X[:, asets[i + 1]], self.beta[asets[i + 1]], tmp_y) @ self.dy_dz
					if real_cand <= cands[i + 1] and real_cand >= cands[i]:
						if np.isclose(cands[i+1], cands[i]).all():
							final_beta = betas[i]
						else:
							first_beta = np.zeros_like(self.betas[-1])
							second_beta = np.zeros_like(self.betas[-1])
							third_beta = np.zeros_like(self.betas[-1])

							first_beta = copy.deepcopy(betas[i+1])
							first_beta[asets[i+1]] +=  play_db_dz_again * (real_cand - cands[i+1]) 
							
							second_beta[asets[i+1]] = self.ja_inverse_model(-1 * self.lmd * duals[i+1], J_As[i+1], Y_z)[0]
							
							slopes = (betas[i + 1] - betas[i])/(cands[i+1] - cands[i])
							third_beta = betas[i] + slopes * (real_cand - cands[i]) 

							first_gap = self.duality_gap(X, Y_z, beta=first_beta, lmd=self.lmd)
							second_gap = self.duality_gap(X, Y_z, beta=second_beta, lmd=self.lmd)	
							third_gap = self.duality_gap(X, Y_z, beta=third_beta, lmd=self.lmd)

							true_gap = self.duality_gap(X, Y_z, beta=true_betas[idx], lmd=self.lmd)
							true_gaps.append(true_gap)
							
							if first_gap < second_gap and first_gap < third_gap:
								final_beta = first_beta
								best_gaps.append(first_gap)
							elif second_gap < first_gap and second_gap < third_gap:
								final_beta = second_beta
								best_gaps.append(second_gap)
							else:
								final_beta = third_beta
								best_gaps.append(third_gap)

						break
			true_primals.append(self.primal(X, Y_z, beta=true_betas[idx]))
			errors.append(self.primal(X, Y_z, beta=final_beta, lmd=self.lmd))
		return np.mean(np.asarray(errors) - np.asarray(true_primals))
		# return np.mean(errors)
	def calc_primals(self, search_space, X, Y):
		betas = self.betas.copy()
		betas.reverse()
		cands = self.cands.copy()
		cands.reverse()
		thetas = self.thetas.copy()
		thetas.reverse()
		duals = self.duals.copy()
		duals.reverse()
		J_As = self.J_As.copy()
		J_As.reverse()
		asets = self.asets.copy()
		asets.reverse()
		Y_z = copy.deepcopy(Y)
		errors = []
		true_gaps = []
		best_gaps = []
		true_primals=[]
		for idx in range(len(search_space)):
			
			real_cand = search_space[idx]
			Y_z[-1] = real_cand
			if real_cand < cands[0]:
				# slopes = (betas[1] - betas[0])/(cands[1] - cands[0])
				# final_beta = betas[0] + slopes * (real_cand - cands[0])
				final_beta = betas[0]
			elif real_cand > cands[-1]:
				# slopes = (betas[-1] - betas[-2])/(cands[-1] - cands[-2])
				# final_beta = betas[-1] + slopes * (real_cand - cands[-1]) 
				final_beta = betas[-1]
			else:
				
				for i in range(len(cands) - 1):
					tmp_y = copy.deepcopy(Y_z)
					tmp_y[-1] = cands[i+1]
					play_db_dz_again = -1 * np.linalg.inv(self.second_beta(X[:, asets[i + 1]], betas[i + 1][asets[i + 1]], tmp_y)) @ self.second_y(X[:, asets[i + 1]], self.beta[asets[i + 1]], tmp_y) @ self.dy_dz
					if real_cand <= cands[i + 1] and real_cand >= cands[i]:
						if np.isclose(cands[i+1], cands[i]).all():
							final_beta = betas[i]
						else:
							first_beta = np.zeros_like(self.betas[-1])
							second_beta = np.zeros_like(self.betas[-1])
							third_beta = np.zeros_like(self.betas[-1])

							first_beta = copy.deepcopy(betas[i+1])
							first_beta[asets[i+1]] +=  play_db_dz_again * (real_cand - cands[i+1]) 
							
							second_beta[asets[i+1]] = self.ja_inverse_model(-1 * self.lmd * duals[i+1], J_As[i+1], Y_z)[0]
							
							slopes = (betas[i + 1] - betas[i])/(cands[i+1] - cands[i])
							third_beta = betas[i] + slopes * (real_cand - cands[i]) 

							first_gap = self.duality_gap(X, Y_z, beta=first_beta, lmd=self.lmd)
							second_gap = self.duality_gap(X, Y_z, beta=second_beta, lmd=self.lmd)	
							third_gap = self.duality_gap(X, Y_z, beta=third_beta, lmd=self.lmd)

							
							
							if first_gap < second_gap and first_gap < third_gap:
								final_beta = first_beta
								best_gaps.append(first_gap)
							elif second_gap < first_gap and second_gap < third_gap:
								final_beta = second_beta
								best_gaps.append(second_gap)
							else:
								final_beta = third_beta
								best_gaps.append(third_gap)
						break
			true_beta, _, _ = self.proximal_func(X, Y_z, self.lmd, self.duality_gap, tol=self.mirror_tol, initial_beta=final_beta)
			true_gaps.append(self.duality_gap(X, Y_z, beta=true_beta))
			true_primals.append(self.primal(X, Y_z, beta=true_beta))
			errors.append(self.primal(X, Y_z, beta=final_beta))
		primal_gaps = np.asarray(errors) - np.asarray(true_primals)
		import pickle
		with open("results/gap_eps_lasso_friedman1.pkl", "wb") as f:
			pickle.dump((best_gaps, primal_gaps, search_space), f)
		return np.mean(np.asarray(errors) - np.asarray(true_primals))
		# return np.mean(errors)

	def get_conf_interval(self, X, Y, alpha):
		ranks, points, _, _ = self.solve_CP(X, Y)

		bounds = []
		num_ele = len(Y)
		for idx in range(len(points)):
			if 1/num_ele * ranks[idx] >= alpha:
				bounds.append(idx)

		if len(bounds) == 0:
			return 0, 0
		top = points[min(bounds)-1] if min(bounds) - 1 >= 0 else points[min(bounds)]
		bottom = points[max(bounds)]
		assert(top >= bottom)
		return bottom,top
			

	def solve_CP(self, X, Y):
		tmp_y = np.concatenate((copy.deepcopy(Y), [0]))
		self.solve(X, Y)
		swaps = []
		tmp_y[-1] = self.cands[0]
		# starting_residuals = abs(tmp_y - X @ self.betas[0])
		# num_greater = len(np.where(starting_residuals > starting_residuals[-1])[0])
		vals = []
		for kink_idx in range(len(self.cands) - 1):
			next_kink = self.cands[kink_idx+1]
			curr_kink = self.cands[kink_idx]

			theta = self.duals[kink_idx]
			J_A = self.J_As[kink_idx]
			aset = self.asets[kink_idx]
			pre_compute_lmd_theta = -1 * self.lmd * theta
			_, precomputed_lhs, precomputed_rhs = self.ja_inverse_model(pre_compute_lmd_theta, J_A, tmp_y)
			for i in range(len(Y) - 1):

				
				def solve_for_z_pos(z):
					tmp_y[-1] = z
					lhs = Y[i] - z
					next_beta = self.ja_inverse_model(pre_compute_lmd_theta, J_A, tmp_y, precomputed_lhs=precomputed_lhs, precomputed_rhs=precomputed_rhs)[0]
					rhs = (X[i, aset] - X[-1, aset]) @ next_beta
					return lhs - rhs

				def solve_for_z_neg(z):
					tmp_y[-1] = z
					lhs = Y[i] + z
					next_beta = self.ja_inverse_model(pre_compute_lmd_theta, J_A, tmp_y, precomputed_lhs=precomputed_lhs, precomputed_rhs=precomputed_rhs)[0]
					rhs = (X[i, aset] + X[-1, aset]) @ next_beta
					return lhs - rhs
				def helper(z, idx):
					tmp_y[-1] = z
					next_beta = self.ja_inverse_model(pre_compute_lmd_theta, J_A, tmp_y, precomputed_lhs=precomputed_lhs, precomputed_rhs=precomputed_rhs)[0]
					return abs(tmp_y[idx] - X[idx, aset] @ next_beta)
				def helper_get_beta(z):
					tmp_y[-1] = z
					next_beta = self.ja_inverse_model(pre_compute_lmd_theta, J_A, tmp_y, precomputed_lhs=precomputed_lhs, precomputed_rhs=precomputed_rhs)[0]
					full_beta = np.zeros(X.shape[1])
					full_beta[aset] = next_beta
					return full_beta
				pos_root = self.do_bisection(solve_for_z_pos, next_kink, curr_kink, tol=1e-4)
				neg_root = self.do_bisection(solve_for_z_neg, next_kink, curr_kink, tol=1e-4)
				
				curr_val = helper(curr_kink, -1)
				basic_val = helper(curr_kink, i)

				
				if curr_val > basic_val:
					sign = "neg"
				else:
					sign = "pos"

				if pos_root > -1000:
					actual_beta = helper_get_beta(pos_root - self.cp_tol)
					swaps.append((pos_root, sign, actual_beta))
				if neg_root > -1000:
					actual_beta = helper_get_beta(neg_root - self.cp_tol)
					swaps.append((neg_root, sign, actual_beta))
		swaps = sorted(swaps, key= lambda x: x[0], reverse=True)

		cand_pointer = 0

		# chicken_vals = als = [(banana_cands[cand_pointer], len(np.where(abs(tmp_y - X @ banana_betas[cand_pointer]) > (abs(tmp_y - X @ banana_betas[cand_pointer]))[-1])[0])) for cand_pointer in range(len(banana_cands))]
		if len(swaps) == 0:
			while cand_pointer < len(self.cands):
				tmp_y[-1] = self.cands[cand_pointer]
				residuals = abs(tmp_y - X @ self.betas[cand_pointer])
				vals.append((self.cands[cand_pointer], len(np.where(residuals > residuals[-1])[0])))
				cand_pointer += 1
		
		for swap in swaps:
			try:
				while cand_pointer < len(self.cands) and (swap[0] < self.cands[cand_pointer] or np.isclose(swap[0], self.cands[cand_pointer])):
					
					tmp_y[-1] = self.cands[cand_pointer]
					# slopes = (self.betas[cand_pointer + 1] - self.betas[cand_pointer])/(self.cands[cand_pointer] - self.cands[cand_pointer + 1])

					# third_beta = self.betas[cand_pointer] + slopes * (swap[0] - self.cands[cand_pointer]) 
					# breakpoint()

					residuals = abs(tmp_y - X @ self.betas[cand_pointer])
					vals.append((self.cands[cand_pointer], len(np.where(residuals > residuals[-1])[0])))
					cand_pointer += 1
					if np.isclose(swap[0], [cand_pointer]):
						continue
				tmp_y[-1] = swap[0] - self.cp_tol
				residuals = abs(tmp_y - X @ swap[2])
				vals.append((swap[0], len(np.where(residuals > residuals[-1])[0])))
				# if swap[1] is "neg":
				# 	vals.append((swap[0], vals[-1][1] + 1))
				# else:
				# 	vals.append((swap[0], vals[-1][1] - 1))
			except:
				breakpoint()
		points, ranks = list(zip(*vals))
		return ranks, points, self.betas, self.cands
	
	def solve(self, X, Y):
		self.betas = []
		self.cands = []
		self.thetas = []
		self.duals = []
		self.J_As = []
		self.asets = []

		self.initialize(X, Y)
		while self.cand > min(Y) * 2:
			status = self.update(X, Y)
			if status == -1:
				break
		if self.cand > min(Y):
			self.update_beta_yz(X, Y)
			

		for idx in range(len(self.cands)):
			if self.cands[idx] < min(Y):
				self.cands = self.cands[:idx]
				self.betas = self.betas[:idx]
				self.duals = self.duals[:idx]
				self.asets = self.asets[:idx]
				self.J_As = self.J_As[:idx]
				self.max_aset = self.max_aset[:idx]
				break

		self.cand = min(Y)
		self.update_beta_yz(X, Y)
		return self.cands, self.betas, self.duals

	def update_beta_yz(self, X, Y):
		
		self.Y_z = np.concatenate((copy.deepcopy(Y), [self.cand]))
		try:
			self.J_A = np.linalg.solve((X[:, self.aset].T @ X[:, self.aset]), X[:, self.aset].T)
		except:
			print("NOT good")
			self.J_A = np.linalg.pinv(X[:, self.aset].T @ X[:, self.aset]) @ X[:, self.aset].T

		self.beta = self.get_mirror(X, self.Y_z, proximal_func=self.proximal_func, mirror_tol=self.mirror_tol)

		if not self.approx_not_used:
			self.aset = np.where(abs(self.beta) > 1e-10)[0].tolist()
			self.naset = np.where(abs(self.beta) < 1e-10)[0].tolist()

			try:
				self.J_A = np.linalg.solve((X[:, self.aset].T @ X[:, self.aset]), X[:, self.aset].T)
			except:
				print("NOT good")
				self.J_A = np.linalg.pinv(X[:, self.aset].T @ X[:, self.aset]) @ X[:, self.aset].T

			self.beta[self.naset] *= 0
		
		Xbeta = X @ self.beta

		grad_f = self.model(Xbeta, self.Y_z)
		self.theta_estim = - grad_f / max(np.linalg.norm(X.T.dot(grad_f), ord=np.inf), self.lmd)
		
		duality_gap = self.duality_gap(X, self.Y_z)
		self.most_current_duality_gap = duality_gap

		self.betas.append(copy.deepcopy(self.beta))
		self.cands.append(copy.deepcopy(self.cand))
		self.duals.append(copy.deepcopy(self.theta_estim))
		self.J_As.append(copy.deepcopy(self.J_A))
		self.asets.append(copy.deepcopy(self.aset))
		self.max_aset.append(np.linalg.norm(self.beta, ord=1))
		self.eta = np.sign(-X.T @ self.model(X @ self.beta, self.Y_z))
		
	def get_lb(self, vec, suggested):
		if len(vec) == 0:
			return max(self.min_Y_val, suggested)
		else:
			return max(max(max(vec), self.min_Y_val), suggested)
	def update(self, X, Y):
		# breakpoint()
		self.update_beta_yz(X, Y)
		self.min_Y_val = self.multiplier * min(Y)
		exits = []
		tmp_y = copy.deepcopy(self.Y_z)
		pre_compute_lmd_theta = -1 * self.lmd * self.theta_estim
		_, precomputed_lhs, precomputed_rhs = self.ja_inverse_model(pre_compute_lmd_theta, self.J_A, tmp_y)
		#Nonzero to zero
		aset_array = np.asarray(self.aset)
		naset_array = np.asarray(self.naset)
		play_db_dz_again = -1 * np.linalg.inv(self.second_beta(X[:, self.aset], self.beta[self.aset], tmp_y)) @ self.second_y(X[:, self.aset], self.beta[self.aset], tmp_y) @ self.dy_dz
		if len(aset_array) != 0:
			def calc_next_cand_nz_full(tmp_z, features):
				tmp_y[-1] = tmp_z
				next_beta_aset =  play_db_dz_again * (tmp_z - self.cands[-1]) + self.beta[self.aset]
				return next_beta_aset[features]
			viable_aset, new_left, new_right = self.do_bisection_checker(calc_next_cand_nz_full, self.min_Y_val, self.cands[-1], self.step_size, len(aset_array))
			good_aset = np.take(self.aset, viable_aset)
			for j in good_aset:
				idx_j = self.aset.index(j)
				def calc_next_cand_nz(tmp_z):
					tmp_y[-1] = tmp_z
					next_beta_aset =  play_db_dz_again * (tmp_z - self.cands[-1]) + self.beta[self.aset]
					return next_beta_aset[idx_j]
				exits.append(self.do_bisection(calc_next_cand_nz, self.get_lb(exits, new_left), new_right, tol=self.bisection_tol))
				
			exits = [ele if ele < self.cand and not np.isclose(ele, self.cand) else -1 * float('inf') for ele in exits]
			best_nz_to_z_index = np.argmax(exits) if len(exits) > 0 else None
			best_new_nz_to_z_cand = exits[best_nz_to_z_index] if len(exits) > 0 else -1 * float("inf")
			best_nz_to_z_index = good_aset[best_nz_to_z_index] if len(exits) > 0 else None
		else:
			best_new_nz_to_z_cand = -1 * float('inf')
		#calc zero to nonzero candidadate
		if len(naset_array) != 0:
			entrances_pos = []
			entrances_neg = []

			tmp_y = copy.deepcopy(self.Y_z)
			_, precomputed_lhs, precomputed_rhs = self.ja_inverse_model(pre_compute_lmd_theta, self.J_A, tmp_y)

			second_der_y = self.second_y(X, self.beta, tmp_y) @ self.dy_dz
			second_der_beta = self.second_beta(X, self.beta, tmp_y)
			starter_curr_df_db =  X.T @ self.model(X @ self.beta, tmp_y)

			help_lhs = second_der_beta[:, self.aset] @ self.beta[self.aset]
			def calc_next_cand_z_pos_full(tmp_z, features):
				tmp_y[-1] = tmp_z
				next_beta_aset =  play_db_dz_again * (tmp_z - self.cands[-1]) + self.beta[self.aset]
				again_expected_val = starter_curr_df_db[naset_array[features]] + second_der_beta[np.ix_(naset_array[features], self.aset)] @ next_beta_aset - help_lhs[naset_array[features]] + second_der_y[naset_array[features]] * (tmp_z - self.cands[-1])
				return -1 * again_expected_val - self.lmd
			def calc_next_cand_z_neg_full(tmp_z, features):
				next_beta_aset =  play_db_dz_again * (tmp_z - self.cands[-1]) + self.beta[self.aset]
				again_expected_val = starter_curr_df_db[naset_array[features]] + second_der_beta[np.ix_(naset_array[features], self.aset)] @ next_beta_aset - help_lhs[naset_array[features]] + second_der_y[naset_array[features]] * (tmp_z - self.cands[-1])
				return again_expected_val - self.lmd
			viable_naset_pos, new_left_pos, new_right_pos = self.do_bisection_checker(calc_next_cand_z_pos_full, self.min_Y_val, self.cands[-1], self.step_size, len(self.naset))
			viable_naset_neg, new_left_neg, new_right_neg = self.do_bisection_checker(calc_next_cand_z_neg_full, self.min_Y_val, self.cands[-1], self.step_size, len(self.naset))

			pos_naset = np.take(self.naset, viable_naset_pos)
			for feat_idx in pos_naset:
				help_lhs = second_der_beta[feat_idx, self.aset] @ self.beta[self.aset]
				def calc_next_cand_z_pos(tmp_z):
					tmp_y[-1] = tmp_z
					next_beta_aset =  play_db_dz_again * (tmp_z - self.cands[-1]) + self.beta[self.aset]
					again_expected_val = starter_curr_df_db[feat_idx] + second_der_beta[feat_idx, self.aset] @ next_beta_aset - help_lhs + second_der_y[feat_idx] * (tmp_z - self.cands[-1])
					return -1 * again_expected_val - self.lmd
				entrances_pos.append(self.do_bisection(calc_next_cand_z_pos, self.get_lb(entrances_pos, new_left_pos), new_right_pos, tol=self.bisection_tol))

			neg_naset = np.take(self.naset, viable_naset_neg)
			for feat_idx in neg_naset:
				help_lhs = second_der_beta[feat_idx, self.aset] @ self.beta[self.aset]
				def calc_next_cand_z_neg(tmp_z):
					next_beta_aset =  play_db_dz_again * (tmp_z - self.cands[-1]) + self.beta[self.aset]
					again_expected_val = starter_curr_df_db[feat_idx] + second_der_beta[feat_idx, self.aset] @ next_beta_aset - help_lhs + second_der_y[feat_idx] * (tmp_z - self.cands[-1])
					return again_expected_val - self.lmd
				entrances_neg.append(self.do_bisection(calc_next_cand_z_neg, self.get_lb(entrances_neg, new_left_neg), new_right_neg, tol=self.bisection_tol))
					
			# breakpoint()
			
			entrances_neg = [ele if ele < self.cand and not np.isclose(ele, self.cand) else -1 * float('inf') for ele in entrances_neg ]
			entrances_pos = [ele if ele < self.cand and not np.isclose(ele, self.cand) else -1 * float('inf') for ele in entrances_pos ]

			neg_best_feat = np.argmax(entrances_neg) if len(entrances_neg) > 0 else None
			neg_best = entrances_neg[neg_best_feat] if len(entrances_neg) > 0 else -1 * float("inf")
			neg_best_feat = neg_naset[neg_best_feat] if len(entrances_neg) > 0 else None

			pos_best_feat = np.argmax(entrances_pos) if len(entrances_pos) > 0 else None
			pos_best = entrances_pos[pos_best_feat] if len(entrances_pos) > 0 else -1 * float("inf")
			pos_best_feat = pos_naset[pos_best_feat] if len(entrances_pos) > 0 else None
		
			if pos_best > neg_best:
				best_z_to_nz_index = pos_best_feat
				best_new_z_to_nz_cand = pos_best
			else:
				best_z_to_nz_index = neg_best_feat
				best_new_z_to_nz_cand = neg_best
		else:
			best_new_z_to_nz_cand = -1 * float('inf')

		self.approx_not_used = True
		# breakpoint()
		if best_new_nz_to_z_cand == -1 * float('inf') and best_new_z_to_nz_cand == -1 * float('inf'):
			return -1
		self.prev_aset = copy.deepcopy(self.aset)
		if best_new_z_to_nz_cand > best_new_nz_to_z_cand:
			self.aset.append(best_z_to_nz_index)
			self.naset.remove(best_z_to_nz_index)
			self.cand = best_new_z_to_nz_cand - self.small_delta
		else:
			self.aset.remove(best_nz_to_z_index)
			self.naset.append(best_nz_to_z_index)
			self.cand = best_new_nz_to_z_cand - self.small_delta

		return 1
		
	def initialize(self, X, Y):
		self.n_samples, self.n_features = X.shape
		self.cand = max(Y)
		
		self.Y_z = np.concatenate((copy.deepcopy(Y), [self.cand]))
		self.dy_dz = np.zeros_like(self.Y_z)
		self.dy_dz[-1] = 1

		self.beta, _, _ = self.proximal_func(X, self.Y_z, self.lmd, self.duality_gap, tol=self.mirror_tol)
		self.naset = np.where(abs(self.beta) < 1e-6)[0]
		self.beta[self.naset] = 0

		duality_gap = self.duality_gap(X, self.Y_z)
		self.most_current_duality_gap = duality_gap

		self.aset = np.squeeze(np.argwhere(self.beta != 0), axis=1).tolist()
		self.naset = list(set(range(self.n_features)) - set(self.aset))
		
		self.eta = np.sign(-X.T @ self.model(X @ self.beta, self.Y_z))
		try:
			self.J_A = np.linalg.solve((X[:, self.aset].T @ X[:, self.aset]), X[:, self.aset].T)
		except:
			self.J_A = np.linalg.pinv(X[:, self.aset].T @ X[:, self.aset]) @ X[:, self.aset].T
			
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dim", type=int, default=10)
	parser.add_argument("--num-examples", type=int, default=20)
	parser.add_argument("--max-val", type=int, default=20)
	parser.add_argument("--min-val", type=int, default=-20)
	parser.add_argument("--loss-type", choices=["assymetric", "lasso", "robust"])

	args = parser.parse_args()

	# diabetes = datasets.load_diabetes()
	# X, Y = diabetes.data, diabetes.target
	# friedman = datasets.make_friedman1()
	# X, Y = friedman[0], friedman[1]
	# transformed_X = X/ X.max(axis=0)
	# transformed_Y = (Y - Y[:-1].mean())/Y[:-1].std()
	# lmd = np.linalg.norm(transformed_X[:-1].T @ lasso_gradient(np.zeros(len(transformed_X[:-1])), transformed_Y[:-1]), ord=np.inf)/20
	# X = transformed_X
	# Y = transformed_Y
	# breakpoint()
	X, Y, beta_true = generate_stdp_dataset(args.dim, args.num_examples, args.min_val, args.max_val)

	lmd = np.linalg.norm(X[:-1].T @ lasso_gradient(np.zeros(args.num_examples), Y[:-1]), ord=np.inf)/20
	# import pickle
	# with open('data/synthetic.pkl', "rb") as f:
	# 	dataset = pickle.load(f)

	# X, Y, true_beta, lmd =  dataset[1000][0]
	# cancer = datasets.load_breast_cancer()
	# X, Y = cancer.data, cancer.target
	# X = X/np.max(X)
	# Y = (Y - Y[:-1].mean())/Y[:-1].std()
	# lmd = np.linalg.norm(X[:-1].T @ lasso_gradient(np.zeros(len(X) - 1), Y[:-1]), ord=np.inf)/20
	# dataset = [X, Y, None, lmd]
	search_space = np.linspace(min(Y), max(Y), 30)
	if args.loss_type == "assymetric":
		assymetric_cp = ZPath(assymetric_loss, assymetric_gradient, inv_assymetric_gradient, prox_assymetric, assymetric_gradient_dbeta, assymetric_gradient_dy, mirror_tol=1e-10, lmd=lmd, ja_inverse_model=ja_inv_assymetric_gradient)
		chosen_cp = assymetric_cp
		cands, betas, duals = assymetric_cp.solve(X, Y[:-1])
		assymetric_cp.calc_primals(search_space, X, Y)
		d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_assymetric, assymetric_cp.duality_gap, lmd,  tol=1e-9)
	elif args.loss_type == "robust":
		robust_cp = ZPath(robust_loss, robust_gradient, inv_robust_gradient, prox_robust, robust_gradient_dbeta, robust_gradient_dy, mirror_tol=1e-10, bisection_tol=1e-4, lmd=lmd, ja_inverse_model=ja_inv_robust_gradient,)
		chosen_cp = robust_cp
		cands, betas, duals = robust_cp.solve(X, Y[:-1])
		robust_cp.calc_primals(search_space, X, Y)
		d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_robust, robust_cp.duality_gap, lmd,  tol=1e-4)

	elif args.loss_type == "lasso":
		lasso_cp = ZPath(lasso_loss, lasso_gradient, inv_lasso_gradient, prox_lasso, lasso_gradient_dbeta, lasso_gradient_dy, mirror_tol=1e-12, bisection_tol=1e-10, lmd=lmd, ja_inverse_model=ja_inv_lasso_gradient)
		chosen_cp = lasso_cp
		cands, betas, duals = lasso_cp.solve(X, Y[:-1])
		lasso_cp.calc_primals(search_space, X, Y)
		search_space = np.linspace(min(Y), max(Y), 30)
		d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals, ranks = WarmDiscretePass(X, Y[:-1], prox_lasso, lasso_cp.duality_gap, lmd, tol=1e-12)

	import pickle
	with open("results/{}_path.pkl".format(args.loss_type), "wb") as f:
		pickle.dump((d_betas, search_space, betas, cands), f)
	plot_path(d_betas, search_space, betas, cands, 'candidates', name="{}_{}".format(args.loss_type, 'z'))
	# # plot_path(d_duals, search_space, duals, cands, 'candidates', name="{}_{}".format(args.loss_type, 'z_duals'))
	plot_single_path(betas, cands, 'candidates', name="{}_{}".format(args.loss_type, 'z_debug'))
	cands.reverse()
	betas.reverse()
	errors = []
	max_aset = 0
	our_gaps = []
	their_gaps = []
	search_space = np.linspace(min(Y), max(Y), 100)
	maybe_duality = chosen_cp.calc_duality_gaps(search_space, X, Y, d_betas)

	# breakpoint()
	for idx in range(len(search_space)):

		real_cand = search_space[idx]
		true_beta = d_betas[idx]
		aset = np.where(abs(true_beta) > 1e-8)[0]
		max_aset = max(max_aset, len(aset))
		if real_cand < cands[0]:
			slopes = (betas[1] - betas[0])/(cands[1] - cands[0])
			final_beta = betas[0] + slopes * (real_cand - cands[0])
			
		elif real_cand > cands[-1]:
			slopes = (betas[-1] - betas[-2])/(cands[-1] - cands[-2])
			final_beta = betas[-1] + slopes * (real_cand - cands[-1]) 
			
		else:
			for i in range(len(cands) - 1):
				if real_cand <= cands[i + 1] and real_cand >= cands[i]:
					if np.isclose(cands[i+1], cands[i]).all():
						final_beta = betas[i]
					else:
						slopes = (betas[i + 1] - betas[i])/(cands[i+1] - cands[i])
						final_beta = betas[i] + slopes * (real_cand - cands[i]) 
					break
		Y_z = copy.deepcopy(Y)
		Y_z[-1] = real_cand


		our_gap =  -1 * np.log10(chosen_cp.duality_gap(X, Y_z, lmd, final_beta))
		their_gap = -1 * np.log10(chosen_cp.duality_gap(X, Y_z, lmd, true_beta))
		our_gaps.append(our_gap)
		their_gaps.append(their_gap)
		import math
		if math.isnan(np.linalg.norm(final_beta - true_beta, ord=np.inf)):
			breakpoint()
		errors.append(np.linalg.norm(final_beta - true_beta, ord=np.inf))
	breakpoint()
	print("Error: {}".format(np.mean(errors)))