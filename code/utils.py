from tkinter import N
from tracemalloc import stop

# from jmespath import search
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import copy
from datetime import datetime

gamma = .5
p = 1.5

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def sigmoid(z):
	return 1/(1 + np.exp(-z))

def generate_log_dataset(dim, num_examples, min_value, max_value):
	X = np.random.random((num_examples + 1, dim)) * (max_value - min_value) + min_value
	beta = np.random.random((dim)) * (max_value - min_value) + min_value

	noise = np.random.normal(0, np.sqrt(max_value - min_value), num_examples + 1)

	Y = X[:num_examples + 1] @ beta + noise

	X /= np.linalg.norm(X, axis=0)
	Y = (Y - Y.mean()) / Y.std()
	Y = sigmoid(Y)
	Y = np.where(Y > 0.5, 1, 0)
	return X, Y, betas

def lasso_loss(answers, Y):
	return .5 * np.power(Y - answers, 2)

def lasso_gradient(answers, Y, precomputed_lhs=None, precomputed_rhs=None):
	return answers - Y

def ja_inv_lasso_gradient(answers, J_A,  Y, precomputed_lhs=None, precomputed_rhs=None):
	if precomputed_lhs is None:
		precomputed_lhs = J_A @ answers
		precomputed_rhs = J_A[:, :-1] @ Y[:-1]

	true_rhs = precomputed_rhs + J_A[:, -1] * Y[-1]
	return precomputed_lhs + (true_rhs), precomputed_lhs, precomputed_rhs

def inv_lasso_gradient(answers, Y):
	return answers + Y

def lasso_gradient_dbeta(X, beta, Y):
	return X.T @ X

def lasso_gradient_dy(X, beta, Y):
	return -1 * X.T



def logit_grad(answers, Y):
	return np.exp(answers)/(1 + np.exp(answers)) - Y

def inv_logit_grad(answers, Y):
	return np.log((answers+ Y)/(1 - answers - Y))

def assymetric_loss(answers, Y):
	return np.exp(gamma * (Y - answers)) - gamma*(Y - answers) - 1

def assymetric_gradient(answers, Y):
	return gamma * (1 - np.exp(gamma * (Y - answers)))

def inv_assymetric_gradient(answers, Y):

	return Y - np.log(1 - answers/gamma)/gamma

def ja_inv_assymetric_gradient(answers, J_A, Y, precomputed_lhs=None, precomputed_rhs=None):
	if precomputed_lhs is None:
		precomputed_lhs = J_A @ np.log(1 - answers/gamma)/gamma
		precomputed_rhs = J_A[:, :-1] @ Y[:-1]
	true_rhs = precomputed_rhs + J_A[:, -1] * Y[-1]
	return true_rhs - precomputed_lhs, precomputed_lhs, precomputed_rhs


def assymetric_gradient_dbeta(X, beta, Y):
	return gamma * gamma * X.T @ np.multiply(X.T, np.exp(gamma * (X @ beta - Y))).T

def assymetric_gradient_dy(X, beta, Y):
	return -1 * gamma * gamma * np.multiply(X.T, np.exp(gamma * (X @ beta - Y)))

def robust_loss(answers, Y):
	return np.power(abs(Y - answers), p)

def robust_gradient(answers, Y):
	res = Y - answers
	return -np.sign(res) * p * np.power(abs(res), p-1)

def inv_robust_gradient(answers, Y):
	return np.sign(answers) * np.power(abs(answers/p), 1/(p-1)) + Y

def ja_inv_robust_gradient(answers, J_A, Y, precomputed_lhs=None, precomputed_rhs=None):
	if precomputed_lhs is None:
		precomputed_lhs = J_A @ (np.sign(answers) * np.power(abs(answers/p), 1/(p-1)))
		precomputed_rhs = J_A[:, :-1] @ Y[:-1]
	true_rhs = precomputed_rhs + J_A[:, -1] * Y[-1]
	return precomputed_lhs + true_rhs, precomputed_lhs, precomputed_rhs


def robust_gradient_dbeta(X, beta, Y):
	return p * (p - 1) * X.T @ np.multiply(X.T, np.power(abs(X @ beta - Y), p-2)).T

def robust_gradient_dy(X, beta, Y):

	return -1 * p * (p-1) * np.multiply(X.T,  np.power(abs(X @ beta - Y), p-2))
	
def generate_dataset(dim, num_examples, min_value, max_value):
	X = np.random.random((num_examples, dim)) * (max_value - min_value) + min_value
	beta = np.random.random((dim)) * (max_value - min_value) + min_value

	noise = np.random.normal(0, np.sqrt(max_value - min_value), num_examples)

	Y = X @ beta + noise

	return X, Y, beta

def prox_assymetric(x, y, lmd, dual_gap_func, tol=1e-8, initial_beta=None, max_iter=10000, **args):
	beta = cp.Variable(x.shape[1])
	objective = cp.Minimize(cp.sum(cp.exp(gamma * (y - x @ beta)) - gamma * (y - x @ beta) - 1) + lmd * cp.norm1(beta))
	problem = cp.Problem(objective)
	if initial_beta is not None:
		# print("using initial beta")a
		beta.value = initial_beta
		problem.solve(solver=cp.SCS, warm_start=True, eps=tol, max_iters=max_iter)
		print(problem.solver_stats.num_iters)

	else:
		problem.solve(solver=cp.SCS, eps=tol, max_iters=max_iter)
	# print("Num iters: {}".format(problem.solver_stats.num_iters))

	if beta.value is None:
		breakpoint()
	# gap = helper.duality_gap(x, y, beta=beta.value)
	grad_f = assymetric_gradient(x @ beta.value, y)
	theta_estim = - grad_f / max(np.linalg.norm(x.T.dot(grad_f), ord=np.inf), lmd)
	return beta.value,  dual_gap_func(x, y, lmd=lmd, beta=beta.value), theta_estim

def prox_robust(x, y, lmd, dual_gap_func,tol=1e-8, initial_beta=None, max_iter=10000, **args):
	# helper = FenCon(robust_loss, robust_grad, inv_robust_grad)
	# helper.lmd = lmd
	beta = cp.Variable(x.shape[1])
	objective = cp.Minimize(cp.sum(cp.abs(y - x @ beta)**1.5) + lmd * cp.norm1(beta))
	problem = cp.Problem(objective)
	if initial_beta is not None:
		beta.value = initial_beta
		# breakpoint()
		problem.solve(solver=cp.SCS, eps=tol, warm_start=True, max_iters=max_iter)
		print(problem.solver_stats.num_iters)
	else: 
		problem.solve(solver=cp.SCS, eps=tol, max_iters=max_iter)
		print(problem.solver_stats.num_iters)

	# print("Num iters: {}".format(problem.solver_stats.num_iters))
	# gap = helper.duality_gap(x, y, beta=beta.value)
	grad_f = robust_gradient(x @ beta.value, y)
	theta_estim = - grad_f / max(np.linalg.norm(x.T.dot(grad_f), ord=np.inf), lmd)
	return beta.value, dual_gap_func(x, y, lmd=lmd, beta=beta.value), theta_estim

def ST(x, mu):
	return np.sign(x) * np.maximum(np.abs(x) - mu, 0.)

def is_kink(beta, prev_beta, tol=1e-10):
	curr_aset = np.where(abs(beta) > tol)[0]
	prev_aset = np.where(abs(prev_beta) > tol)[0]
	if len(curr_aset) == len(prev_aset) and (curr_aset == prev_aset).all():
		return False
	else:
		return True

def prox_lasso(X, y, lmd, dual_gap_func, tol=1e-16, max_iter=10000, use_path=False, early_stop=False, initial_beta=None):
	if initial_beta is None:
		beta = np.zeros(X.shape[1])
	else:
		beta = initial_beta
	mu = 1 / np.linalg.norm(X, ord=2) ** 2
	zeta = 0.5
	gaps = []
	print(max_iter)
	print(tol)
	for n_iter in range(max_iter):

		res = y - X.dot(beta)
		grad = -X.T.dot(res)
		beta = ST(beta - mu * grad, lmd * mu)

		# Compute gap target
		gap = dual_gap_func(X, y, lmd=lmd, beta=beta)
		gaps += [gap]
		# print(gap)
		# print(beta)
		if gap <= tol:
			# print("converged:", gap <= tol)
			grad_f = lasso_gradient(X @ beta, y)
			theta_estim = - grad_f / max(np.linalg.norm(X.T.dot(grad_f), ord=np.inf), lmd)
			print("Niter: {}".format(n_iter))
			return beta, gap, theta_estim
	print("not converged: gap: {} niter: {}".format(gap, n_iter))
	# print("converged:", gap <= tol)
	# print(gap)

	grad_f = lasso_gradient(X @ beta, y)
	theta_estim = - grad_f / max(np.linalg.norm(X.T.dot(grad_f), ord=np.inf), lmd)
	return beta, gap, theta_estim

def plot_path(dbetas, dkinks, betas, kinks, xaxis, name=None):
	dbetas = np.asarray(dbetas)
	betas = np.asarray(betas)
	now = datetime.now()
	cmap = get_cmap(dbetas.shape[1])
	stopping_idx = -1
	# for idx, kink in enumerate(kinks):
	# 	if kink < min(dkinks):
	# 		stopping_idx = idx
	# 		break
	plt.clf()
	# breakpoint()
	for i in range(dbetas.shape[1]):
		plt.plot(dkinks, dbetas[:, i], '-', kinks, betas[:, i], '--', label="{}".format(i), color=cmap(i))
		# plt.plot(, label="O{}".format(i))
	time_name = now.strftime("%m_%d_%H:%M")
	plt.xlabel(xaxis)
	plt.legend()
	plt.title(name)
	if name is None:
		plt.savefig("code/images/paths/{}.png".format(time_name))
	else:
		plt.savefig("code/images/paths/{}_{}.png".format(name, time_name))

def plot_single_path(betas, kinks, xaxis, name=None):
	betas = np.asarray(betas)
	kinks = np.asarray(kinks)
	now = datetime.now()
	cmap = get_cmap(betas.shape[1] + 1)
	plt.clf()
	for i in range(betas.shape[1]):
		plt.plot(kinks, betas[:, i], '-', label="{}".format(i), color=cmap(i + 1))
		# plt.plot(, label="O{}".format(i))
	time_name = now.strftime("%m_%d_%H:%M")
	plt.xlabel(xaxis)
	plt.legend()
	plt.title(name)
	if name is None:
		plt.savefig("code/images/paths/{}.png".format(time_name))
	else:
		plt.savefig("code/images/paths/{}_{}.png".format(name, time_name))

def plot_really_again(betas, search_space, second_beta, second_y,  X, Y, lmd, xaxis, name=None):
	betas = np.asarray(betas)
	search_space = np.asarray(search_space)
	all_vals = np.zeros((len(search_space), betas[0].shape[0]))
	other_all_vals = np.zeros((len(search_space), betas[0].shape[0]))
	prev_aset = np.where(abs(betas[0]) >= 1e-10)[0]
	cmap = get_cmap(betas[0].shape[0] + 1)
	# all_vals = np.zeros((len(search_space), betas[0].shape[0]))
	other_all_vals = np.zeros((len(search_space), betas[0].shape[0]))
	now = datetime.now()
	for idx, val in enumerate(search_space[:-10]):
		other_idx = idx + 10
		other_val = search_space[other_idx]
		tmp_y = np.concatenate((copy.deepcopy(Y), [val]))
		other_tmp_y = np.concatenate((copy.deepcopy(Y), [other_val]))
		aset = np.where(abs(betas[idx]) >= 1e-8)[0]
		R = tmp_y -  X @ betas[idx]
		dy_dz = np.zeros_like(tmp_y)
		dy_dz[-1] = 1
		try:
			db_dz = -1 * np.linalg.inv(second_beta(X[:, aset], betas[idx, aset], tmp_y)) @ second_y(X[:, aset], betas[idx, aset], tmp_y) @ dy_dz
		except:
			breakpoint()
		true_db_dz = (betas[other_idx] - betas[idx])/(other_val - val)
		other_all_vals[other_idx] = np.zeros_like(true_db_dz)
		other_all_vals[idx, aset] = db_dz
		all_vals[idx] = true_db_dz
		next_aset = np.where(abs(betas[other_idx]) >= 1e-10)[0]
		# if not (abs(all_vals[idx, aset] - other_all_vals[idx, aset]) < .02).all() and len(aset) == len(next_aset) and (aset == next_aset).all():
		# 	breakpoint()
		

	for i in range(all_vals.shape[1]):
		# plt.plot(search_space, all_vals[:, i]/other_all_vals[:, i], ':', )
		plt.plot(search_space, all_vals[:, i], ':', search_space, other_all_vals[:, i], '--', label="{}".format(i), color=cmap(i))
	time_name = now.strftime("%m_%d_%H:%M")
	plt.xlabel(xaxis)
	plt.legend()
	plt.title(name)
	if name is None:
		plt.savefig("images/paths/{}.png".format(time_name))
	else:
		plt.savefig("images/paths/{}_{}.png".format(name, time_name))



def plot_again(betas, search_space, grad_f, inv_grad_f, one_further, X, Y, lmd, xaxis, name=None):
	betas = np.asarray(betas)
	search_space = np.asarray(search_space)
	now = datetime.now()
	final_vals = []
	all_vals = np.zeros((len(search_space), betas[0].shape[0]))
	other_all_vals = np.zeros((len(search_space), betas[0].shape[0]))
	prev_aset = np.where(abs(betas[0]) >= 1e-10)[0]
	cmap = get_cmap(betas[0].shape[0] + 1)
	for idx, val in enumerate(search_space[:-10]):
		other_idx = idx + 10
		other_val = search_space[other_idx]
		tmp_y = np.concatenate((copy.deepcopy(Y), [val]))
		other_tmp_y = np.concatenate((copy.deepcopy(Y), [other_val]))
		aset = np.where(abs(betas[idx]) >= 1e-8)[0]
		if len(aset) != len(prev_aset) or not (aset == prev_aset).all():
			print(search_space[idx])
		prev_aset = aset
		naset = np.where(abs(betas[idx]) <= 1e-10)[0]
		all_vals[idx] = np.squeeze(X.T @ grad_f(np.squeeze(X @ betas[idx].T), tmp_y)/lmd) -  np.squeeze(X.T @ grad_f(np.squeeze(X @ betas[other_idx].T), other_tmp_y)/lmd)

		sigma_aset = X[:, aset].T @ X[:, aset]
		# eta_aset = np.linalg.solve(sigma_aset, X[-1, aset])

		sigma_x = np.linalg.solve(sigma_aset, X[:, aset].T)
		current_theta_time_lambda = grad_f(X @ betas[idx], tmp_y)

		curr_theta = grad_f(X @ betas[idx], tmp_y)/lmd
		next_theta = grad_f(X @ betas[other_idx], other_tmp_y)/lmd

		desired_val = X[:, aset].T @ (inv_grad_f(next_theta * lmd, other_tmp_y) - inv_grad_f(curr_theta * lmd, tmp_y))
		expected_val = X[:, aset].T @ (other_tmp_y - tmp_y +  np.multiply(one_further(lmd * curr_theta, tmp_y), next_theta - curr_theta))
		# print(expected_val - desired_val)



		approx_next_beta = copy.deepcopy(betas[idx])
		approx_next_beta[aset] += sigma_x @ (inv_grad_f(current_theta_time_lambda, other_tmp_y) - inv_grad_f(current_theta_time_lambda, tmp_y))
		other_all_vals[idx] = np.zeros_like(betas[idx])
		other_all_vals[idx] = np.squeeze(X.T @ grad_f(np.squeeze(X @ betas[idx].T), tmp_y)/lmd) -  np.squeeze(X.T @ grad_f(np.squeeze(X @ approx_next_beta.T), other_tmp_y)/lmd)

		# # print(min(sigma_x @ (inv_grad_f(current_theta_time_lambda, other_tmp_y) - inv_grad_f(current_theta_time_lambda, tmp_y))))
		# diff_in_vs_naset = X[:, naset].T @ (grad_f(X @ approx_next_beta, other_tmp_y) - grad_f(X @ betas[idx], tmp_y))

		# # diff_in_t_val = search_space[idx] - search_space[other_idx]
		# # diff_in_t_vec = np.zeros_like(Y)
		# # diff_in_t_vec[-1] = diff_in_t_val
		
		# # gamma_ac = X[-1, naset].T - X[:, naset].T @ X[:, aset] @ eta_aset
		# other_all_vals[idx] = np.zeros_like(betas[idx])
		# other_all_vals[idx][naset] = diff_in_vs_naset
		try:
			if (abs(other_all_vals[idx] - all_vals[idx]) > .05).any() and (aset == np.where(abs(betas[other_idx]) > 1e-10)[0]):
				breakpoint()
		except:
			if (abs(other_all_vals[idx] - all_vals[idx]) > .05).any() and (aset == np.where(abs(betas[other_idx]) > 1e-10)[0]).all():
				breakpoint()

	for i in range(all_vals.shape[1]):
		# plt.plot(search_space, all_vals[:, i]/other_all_vals[:, i], ':', )
		plt.plot(search_space, all_vals[:, i], ':', search_space, other_all_vals[:, i], '*', label="{}".format(i), color=cmap(i))
	time_name = now.strftime("%m_%d_%H:%M")
	plt.xlabel(xaxis)
	plt.legend()
	plt.title(name)
	if name is None:
		plt.savefig("images/paths/{}.png".format(time_name))
	else:
		plt.savefig("images/paths/{}_{}.png".format(name, time_name))

def plot_ranks(discrete_ranks, search_space, our_ranks, our_space, name=None):
	plt.plot(search_space,discrete_ranks, label="Discrete")
	plt.step(our_space, our_ranks, label="Ours", where="post")
	plt.legend()
	plt.xlabel("Z")
	plt.ylabel("Ranks")
	plt.title(name)
	now = datetime.now()
	time_name = now.strftime("%m_%d_%H:%M")
	if name is None:
		plt.savefig("code/images/paths/{}.png".format(time_name))
	else:
		plt.savefig("code/images/paths/{}_{}.png".format(name, time_name))