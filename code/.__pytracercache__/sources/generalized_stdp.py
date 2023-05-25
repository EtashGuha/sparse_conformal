import copy
import numpy as np
import sklearn
import scipy
import argparse 
from lassoreg_path import lasso_path, primal, dual
from sklearn import linear_model

def generate_stdp_dataset(dim, num_examples, min_value, max_value):
	X = np.random.random((num_examples + 1, dim)) * (max_value - min_value) + min_value
	beta = np.random.random((dim)) * (max_value - min_value) + min_value

	noise = np.random.normal(0, np.sqrt(max_value - min_value), num_examples)

	Y = X[:num_examples] @ beta + noise

	return X, Y, beta

class FenCon():
	def __init__(self, model, inverse_model, lmd=0):
		self.model = model 
		self.inverse_model = inverse_model
		self.lmd = lmd

	def solve(self, X, Y):
		self.betas = []
		self.cands = []

		self.initialize(X, Y)
		while not np.isclose(self.cand, 1e-6):
			status = self.update(X, Y)
			print(self.aset)
			if status == -1:
				break


		self.betas.append(copy.deepcopy(self.beta))
		self.cands.append(self.cand)
		return self.betas, self.cands

	def update(self, X, Y):
		print("hello")
		self.betas.append(copy.deepcopy(self.beta))
		self.cands.append(self.cand)

		self.Y_z = np.concatenate((copy.deepcopy(Y), [self.cand]))
		try:
			self.J_A = np.linalg.solve(X[:, self.aset].T @ X[:, self.aset], X[:, self.aset].T)
			pseudo_temp = np.linalg.pinv(X[:, self.aset] @ X[:, self.aset].T)
			self.C_A = (pseudo_temp @ X[:, self.aset]).dot(self.eta[self.aset])
		except:
			breakpoint()
		self.beta[self.aset] = self.J_A @ self.inverse_model(-1 * self.lmd * self.C_A, self.Y_z)
		self.beta[self.naset] *= 0

		self.eta = np.sign(-X.T @ self.model(X @ self.beta, self.Y_z))

		primal_val = primal(X, self.Y_z, self.beta, self.lmd)
		dual_val = dual(X, self.Y_z, self.beta, self.lmd)

		if primal_val - dual_val > 1e-5:
			model = sklearn.linear_model.LassoLars(alpha=self.lmd/X.shape[0], fit_intercept=False)
			model.fit(X, self.Y_z)
			other_beta = model.coef_
			primal_val_other = primal(X, self.Y_z, self.beta, self.lmd)
			dual_val_other = dual(X, self.Y_z, self.beta, self.lmd)
			other_gap = primal_val_other - dual_val_other
			
			our_gap = primal_val - dual_val

			# print("DIFFERENCE: {}".format(our_gap - other_gap))
			# breakpoint()
		print(primal_val - dual_val)
		
		best_nz_to_z_index = None
		best_new_nz_to_z_cand = -1 * float('inf')
		#Nonzero to zero
		for j in range(self.n_features):
			if j not in self.aset:
				continue

			idx_j = self.aset.index(j)
			curr_cand = None

			def calc_next_cand_nz(tmp_z):
				tmp_y = copy.deepcopy(self.Y_z)
				tmp_y[-1] = tmp_z
				temp_beta = np.zeros_like(self.beta)
				temp_beta[self.aset] = self.J_A @ self.inverse_model(-1 * self.lmd * self.C_A, tmp_y)
				return temp_beta[j]

			if not np.sign(calc_next_cand_nz(-1 * self.cand)) == np.sign(calc_next_cand_nz(self.cand)):
				curr_cand = scipy.optimize.bisect(calc_next_cand_nz, -1 * self.cand, self.cand)
			else:
				curr_cand = -1 * float('inf')

			if abs(curr_cand - self.cand) < 1e-5:
				curr_cand = -1 * float('inf')

			if curr_cand is not None and curr_cand > best_new_nz_to_z_cand:
				best_nz_to_z_index = j
				best_new_nz_to_z_cand = curr_cand
			
		best_z_to_nz_index = None
		best_new_z_to_nz_cand = -1 * float('inf')
		#calc zero to nonzero candidadate
		for feat_idx in range(self.n_features):
			if feat_idx not in self.naset:
				continue

			def calc_next_cand_z_pos(tmp_z):
				tmp_y = copy.deepcopy(self.Y_z)
				tmp_y[-1] = tmp_z
				temp_beta = np.zeros_like(self.beta)
				temp_beta[self.aset] = self.J_A @ self.inverse_model(-1 * self.lmd * self.C_A, tmp_y)
				return (-X[:, feat_idx].T @ self.model(X[:, self.aset] @ temp_beta[self.aset], tmp_y)) - self.lmd
			
			def calc_next_cand_z_neg(tmp_z):
				tmp_y = copy.deepcopy(self.Y_z)
				tmp_y[-1] = tmp_z
				temp_beta = np.zeros_like(self.beta)
				temp_beta[self.aset] = self.J_A @ self.inverse_model(-1 * self.lmd * self.C_A, tmp_y)
				return (X[:, feat_idx].T @ self.model(X[:, self.aset] @ temp_beta[self.aset], tmp_y)) - self.lmd
			
			curr_cand = None
			
			if not np.sign(calc_next_cand_z_pos(-1 * self.cand)) == np.sign(calc_next_cand_z_pos(self.cand)):
				curr_cand_pos = scipy.optimize.bisect(calc_next_cand_z_pos, -1 * self.cand, self.cand)
			else:
				curr_cand_pos = -1 * float('inf')
			if not np.sign(calc_next_cand_z_neg(-1 * self.cand)) == np.sign(calc_next_cand_z_neg(self.cand)):
				curr_cand_neg = scipy.optimize.bisect(calc_next_cand_z_neg, -1 * self.cand, self.cand)
			else:
				curr_cand_neg = -1 * float('inf')
			curr_cand = max(curr_cand_neg, curr_cand_pos)

			if abs(curr_cand - self.cand) < 1e-5:
				curr_cand = -1 * float('inf')
			if curr_cand is not None and curr_cand > best_new_nz_to_z_cand:
				best_z_to_nz_index = feat_idx
				best_new_z_to_nz_cand = curr_cand

		prev_aset = copy.deepcopy(self.aset)
		if best_new_nz_to_z_cand == -1 * float('inf') and best_new_z_to_nz_cand == -1 * float('inf'):
			return -1
		if best_new_z_to_nz_cand > best_new_nz_to_z_cand:
			self.aset.append(best_z_to_nz_index)
			self.naset.remove(best_z_to_nz_index)
			self.cand = best_new_z_to_nz_cand
		else:
			self.aset.remove(best_nz_to_z_index)
			self.naset.append(best_nz_to_z_index)
			self.cand = best_new_nz_to_z_cand

		if self.aset == prev_aset:
			breakpoint()
		return 1

	def initialize(self, X, Y):
		self.n_samples, self.n_features = X.shape

		self.cand = max(Y)

		self.Y_z = np.concatenate((copy.deepcopy(Y), [self.cand]))

		model = sklearn.linear_model.LassoLars(alpha=self.lmd/X.shape[0], fit_intercept=False)
		model.fit(X, self.Y_z)

		self.beta = model.coef_

		
		primal_val = primal(X, self.Y_z, self.beta, self.lmd)
		dual_val = dual(X, self.Y_z, self.beta, self.lmd)
		gap = primal_val - dual_val

		print(gap)
		breakpoint()

		self.aset = np.squeeze(np.argwhere(self.beta != 0), axis=1).tolist()
		self.naset = list(set(range(self.n_features)) - set(self.aset))
		
		self.cands.append(copy.deepcopy(self.cand))
		self.betas.append(copy.deepcopy(self.beta))

		self.eta = np.sign(-X.T @ self.model(X @ self.beta, self.Y_z))
		self.J_A = np.linalg.solve(X[:, self.aset].T @ X[:, self.aset], X[:, self.aset].T)
		pseudo_temp = np.linalg.pinv(X[:, self.aset] @ X[:, self.aset].T)
		self.C_A = (pseudo_temp @ X[:, self.aset]).dot(self.eta[self.aset])

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dim", type=int, default=50)
	parser.add_argument("--num-examples", type=int, default=30)
	parser.add_argument("--max-val", type=int, default=100)
	parser.add_argument("--min-val", type=int, default=-100)

	args = parser.parse_args()
	X, Y, beta_true = generate_stdp_dataset(args.dim, args.num_examples, args.min_val, args.max_val)
	
		
	def gradient(answers, labels):
		return answers - labels
	
	def inv_gradient(answers, labels):
		return answers + labels
	
	
	m = FenCon(gradient, inv_gradient, lmd=1)

	betas, cands = m.solve(X, Y)

	print(cands)