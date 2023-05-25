import numpy as np
from code.utils import plot_single_path, prox_assymetric, plot_path, prox_lasso
from code.discretized_z_path import WarmDiscretePass

class ridge:
	""" Ridge estimator.
	"""

	def __init__(self, lmd=0.1):

		self.lmd = lmd
		self.hat = None
		self.hatn = None

	def fit(self, X, y):

		if self.hat is None:
			G = X.T.dot(X) + self.lmd * np.eye(X.shape[1])
			self.hat = np.linalg.solve(G, X.T)

		if self.hatn is None:
			y0 = np.array(list(y[:-1]) + [0])
			self.hatn = self.hat.dot(y0)

		self.beta = self.hatn + y[-1] * self.hat[:, -1]

	def predict(self, X):

		return X.dot(self.beta)

	def conformity(self, y, y_pred):

		return 0.5 * np.square(y - y_pred)


class regressor:

	def __init__(self, model=None, s_eps=0., conform=None):

		self.model = model
		self.coefs = []
		self.s_eps = s_eps
		self.conform = conform

	def fit(self, X, y):

		refit = True

		for t in range(len(self.coefs)):

			if self.s_eps == 0:
				break

			if abs(self.coefs[t][0] - y[-1]) <= self.s_eps:
				self.beta = self.coefs[t][1].copy()
				refit = False
				break

		if refit:
			self.beta = self.model.fit(X, y)
			if self.s_eps != 0:
				self.coefs += [[y[-1], self.beta.copy()]]

	def predict(self, X):
		if len(X.shape) == 1:
			X = X.reshape(1, -1)

		return self.model.predict(X)

	def conformity(self, y, y_pred):

		if self.conform is None:
			return np.abs(y - y_pred)

		else:
			return self.conform(y, y_pred)

class proximal_regressor:

	def __init__(self, proximal_func, lmd, dual_gap_func, tol, max_iter, max_is_eps=0.,  s_eps=0., conform=None):
		self.lmd = lmd
		self.proximal_func = proximal_func
		self.dual_gap_func = dual_gap_func
		self.tol = tol
		self.max_iter = max_iter
		self.coefs = []
		self.s_eps = s_eps
		self.conform = conform

	def fit(self, X, y):

		refit = True

		for t in range(len(self.coefs)):

			if self.s_eps == 0:
				break

			if abs(self.coefs[t][0] - y[-1]) <= self.s_eps:
				self.beta = self.coefs[t][1].copy()
				refit = False
				break

		if refit:
			self.beta, _, _ = self.proximal_func(X, y, self.lmd, self.dual_gap_func, tol=self.tol, max_iter=self.max_iter)
			if self.s_eps != 0:
				self.coefs += [[y[-1], self.beta.copy()]]

	def predict(self, X):
		if len(X.shape) == 1:
			X = X.reshape(1, -1)

		return X @ self.beta

	def conformity(self, y, y_pred):

		if self.conform is None:
			return np.abs(y - y_pred)

		else:
			return self.conform(y, y_pred)


class FenConCP:

	def __init__(self, model=None, s_eps=0., conform=None, name="cp"):
		self.model = model
		self.conform = conform
		self.trained_already = False
		self.name = name

	def fit(self, X, y):
		
		if not self.trained_already:
			print("Arg")
			self.cands, self.betas, self.duals = self.model.solve(X, y[:-1])
			# d_betas, d_gaps, d_kinks, d_kink_betas, search_space, d_duals = WarmDiscretePass(X, y[:-1], prox_lasso,self.model.duality_gap, self.model.lmd)
			# plot_path(d_betas, search_space, self.betas, self.cands, 'candidates', name="{}_{}".format("lasso", 'z'))
			# plot_single_path(self.betas, self.cands, 'candidates', name="{}_{}".format("lasso", 'z_debug'))
			# plot_single_path(self.betas, self.cands, "candidates", self.name)
			self.cands.reverse()
			self.betas.reverse()
			self.trained_already = True

		if y[-1] < self.cands[0]:
			slopes = (self.betas[1] - self.betas[0])/(self.cands[1] - self.cands[0])
			self.beta = self.betas[0] + slopes * (y[-1] - self.cands[0])
			return 
		elif y[-1] > self.cands[-1]:
			slopes = (self.betas[-1] - self.betas[-2])/(self.cands[-1] - self.cands[-2])
			self.beta = self.betas[-1] + slopes * (y[-1] - self.cands[-1]) 
			return
		else:
			for i in range(len(self.cands) - 1):
				if y[-1] < self.cands[i + 1] and y[-1] > self.cands[i]:
					slopes = (self.betas[i + 1] - self.betas[i])/(self.cands[i+1] - self.cands[i])
					self.beta = self.betas[i] + slopes * (y[-1] - self.cands[i]) 
					return
		breakpoint()

	def predict(self, X):

		if len(X.shape) == 1:
			X = X.reshape(1, -1)

		return X.dot(self.beta)

	def conformity(self, y, y_pred):

		if self.conform is None:
			return np.abs(y - y_pred)

		else:
			return self.conform(y, y_pred)