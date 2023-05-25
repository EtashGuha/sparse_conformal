import numpy as np
from sklearn.datasets import make_regression
from rootcp.tools import ridgeCP, set_style
import matplotlib.pyplot as plt
import time
from scipy.special import expit
from rootcp import rootCP, models
import pandas as pd


plt.rcParams["text.usetex"] = True
set_style()

# random_state = np.random.randint(1000)
random_state = 825  # Used to generate the image (feel free to change it)

print("random_state", random_state)
alpha = 0.1

n_samples, n_features = (300, 50)
X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       random_state=random_state)

X /= np.linalg.norm(X, axis=0)
y = (y - y.mean()) / y.std()

X = np.asfortranarray(X)
y = np.asfortranarray(y)

# lmd_max = np.log(n_features)
# lmd = lmd_max / 50.

X_ = X[:-1]
y_ = y[:-1]
norm_beta_ols = np.linalg.norm(np.linalg.solve(X_.T.dot(X_), X_.T).dot(y_))
lmd_ols = n_features / (norm_beta_ols ** 2)
# lmd_min = lmd_ols / 20
lmd = lmd_ols

print(lmd_ols, np.log(n_features) / 50)

gammas = [20, 50, 100, 250, 500, 1000, 10000]


def model_fit(X, obs):

    return hat.dot(obs)


def conformity_measure(obs, pred_obs):

    return np.square(obs - pred_obs)


def rank(u, index=-1):

    return np.sum(u - u[index] <= 0)


def sigmoid(x, gamma):

    sig = 1 - expit(gamma * x)

    return sig


def srank(u, index=-1, gamma=50):

    return np.sum(sigmoid(u - u[index], gamma))


def pvalue(z, gamma=50):

    y_z = np.array(list(y[:-1]) + [z])
    beta = model_fit(X, y_z)
    scores = conformity_measure(y_z, X.dot(beta))
    srank_z = srank(scores, -1, gamma=gamma)
    spi = 1 - srank_z / n_samples

    return spi


# tic = time.time()
hat = np.linalg.solve(X.T.dot(X) + lmd * np.eye(n_features), X.T)
# numerical_interval = conf_pred(X, y[:-1], model_fit, conformity_measure)
# time_numerical = time.time() - tic
# print("time_numerical is", time_numerical)


def plot_cp(interval, hat_y, p_values, c1="k", c2="red"):

    size = hat_y.shape[0]
    for i in range(1, size - 2):
        y_range = np.linspace(hat_y[i], hat_y[i + 1], 100)

        if i == size - 3:
            label = r"$\pi$"
        else:
            label = None

        plt.plot(y_range, y_range.shape[0] * [p_values[i]], color=c1,
                 label=label)
        lo = interval.lower
        up = interval.upper

        if (hat_y[i] <= lo) and (lo <= hat_y[i + 1]):
            l_val = p_values[i]

        if (hat_y[i] <= up) and (up <= hat_y[i + 1]):
            u_val = p_values[i]

    plt.scatter(lo, l_val, c=c2, marker="x", s=80)
    plt.scatter(up, u_val, c=c2, marker="x", s=80)


tic = time.time()
interval, hat_y, p_values = ridgeCP(X, y[:-1], lmd, alpha)
time_exact = time.time() - tic
print("time_exact is", time_exact)

z_min, z_max = np.min(y[:-1]), np.max(y[:-1])
y_0 = np.array(list(y[:-1]) + [0.])
beta = hat.dot(y_0)
z_pred = X[-1, :].dot(beta)

min_z, max_z = min(hat_y[1:-1]), max(hat_y[1:-1])
range_z = np.linspace(min_z, max_z, 100)

plt.figure()
plot_cp(interval, hat_y, p_values)

for gamma in gammas:
    pi_z = np.array([pvalue(z, gamma=gamma) for z in range_z])
    plt.plot(range_z, pi_z)
    # c="g", label=r"$\pi(\cdot, \gamma)$")

plt.hlines(alpha, z_min, z_max, colors="k", lw=0.75)
plt.xlim((min(hat_y[1:-1]), max(hat_y[1:-1])))
plt.xlabel("Candidate z")
plt.ylabel(r"$\pi(z)$" + " and " + r"$\pi(\cdot, \gamma)$")
plt.grid(None)
plt.legend()
plt.tight_layout()
plt.savefig("smoothing_pi.pdf", format="pdf")
plt.show()


plt.figure()
range_z = np.linspace(-0.35, 0.35, 100)
ind_z = np.array([int(z <= 0) for z in range_z])
plt.plot(range_z, ind_z, label=r"$1_{\cdot \leq 0}$", c="k")

for ig, gamma in enumerate(gammas):
    sig_z = np.array([1 - expit(gamma * z) for z in range_z])
    plt.plot(range_z, sig_z, label=r"$\gamma =$" + str(gamma))

plt.xlabel("z")
plt.ylabel(r"$\phi_{\gamma}(z)=\frac{\exp^{-\gamma z}}{1 + \exp^{-\gamma z}}$")
plt.grid(None)
plt.legend()
plt.tight_layout()
plt.savefig("smoothing_ind.pdf", format="pdf")
plt.show()


model = models.ridge(lmd=lmd)
algos = ['bisect', 'toms748', 'ridder', 'brentq', 'brenth']
colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442"]
set_pow = [3, 6, 12, 16]
tols = np.power(0.1, set_pow)
res = np.zeros((len(tols), len(algos)))


gammas = [100, 10000]

for gamma in gammas:
    for i_tol, tol in enumerate(tols):
        for i_a, algo in enumerate(algos):
            _, res[i_tol, i_a] = rootCP.conformalset(X, y[:-1], model,
                                                     gamma=gamma, tol=tol,
                                                     algo=algo, stat=True)

    index = [r"$10^{-%s}$" % str(a) for a in set_pow]
    df = pd.DataFrame(res, columns=algos, index=index)
    df.plot.bar(rot=0, color=colors)
    plt.xlabel("Tolerance (root)")
    plt.ylabel(r"$\#$ " + "model fit " + r"$(\gamma=%s$)" % str(gamma))
    plt.grid(None)
    plt.tight_layout()
    plt.savefig("smoothing_gamma" + str(gamma) + ".pdf", format="pdf")
    plt.show()


gammas = np.linspace(10, 10000, 1000)
nfits = np.zeros((len(gammas), len(algos)))
lengths = np.zeros((len(gammas), len(algos)))

tols = [1e-4, 1e-12]

for tol in tols:
    for i_g, gamma in enumerate(gammas):
        for i_a, algo in enumerate(algos):
            output = rootCP.conformalset(X, y[:-1], model, gamma=gamma,
                                         tol=tol, algo=algo, stat=True)

            nfits[i_g, i_a] = output[1]
            cp = output[0]
            lengths[i_g, i_a] = cp[1] - cp[0]

    plt.figure()
    for i_a, algo in enumerate(algos):
        plt.plot(gammas, nfits[:, i_a], label=algo, c=colors[i_a])

    tol_pow = int(np.log10(tol))
    plt.ylabel(r"$\#$ " + "model fit " + r"$(tol = 10^{%s}$)" % str(tol_pow))
    plt.xlabel(r"$\gamma$")
    plt.legend()
    plt.grid(None)
    plt.tight_layout()
    plt.savefig("smoothing_movegamma_tol" + str(tol_pow) + ".pdf",
                format="pdf")
    plt.show()

# plt.figure()
# for i_a, algo in enumerate(algos):
#     plt.plot(gammas, lengths[:, i_a], label=algo, c=colors[i_a])

# plt.ylabel(r"length " + r"$(tol = 10^{-4}$)")
# plt.xlabel(r"$\gamma$")
# plt.legend()
# plt.grid(None)
# plt.tight_layout()
# plt.show()
