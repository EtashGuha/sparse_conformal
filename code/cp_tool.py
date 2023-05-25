# from code.generalized_z_path import assymetric_grad, inv_assymetric_grad, inv_robust_grad, robust_grad
import numpy as np
from generalized_z_path import FenCon, lasso_loss, lasso_gradient, inv_lasso_gradient, assymetric_loss, assymetric_grad, inv_assymetric_grad, robust_loss, robust_grad, inv_robust_grad
class cpregressor():
	def __init__(self, invgradf, lmd):
		self.invgradf = invgradf
		self.lmd = lmd

	def fit(self, X, y):
		m = FenCon(lasso_loss, lasso_gradient, inv_lasso_gradient)
		betas, cands = m.solve(X, y[:-1])


		z = y[-1]
		if z > cands[0]:
			currb = betas[0] 
		elif z < cands[-1]:
			currb = betas[-1]
		else:
			for idx, cand in enumerate(cands):
				
				if cand > z:
					currb = betas[idx]
					break
	
		
		self.aset = np.argwhere(abs(currb) > 1e-10)
		pseudo_temp = np.linalg.pinv(X[:, self.aset] @ X[:, self.aset].T)
		Pi_A = (pseudo_temp @ X[:, self.aset]).dot(self.eta[self.aset])
		Pi_At = np.linalg.solve(X[:, self.aset].T @ X[:, self.aset], X[:, self.aset].T)
		v_a = np.sign(currb)

		self.beta = Pi_A @ self.invgradf(-self.lmd * Pi_At * v_a, y)

	def predict(self, X):
		return X.dot(self.beta)
		
	def conformity(self, y, y_pred):
		return 0.5 * np.square(y - y_pred)

