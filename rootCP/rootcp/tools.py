import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from rootCP.rootcp import models
from sklearn.datasets import make_regression
from sklearn import datasets
import intervals


def oracleCP(X, y, model, alpha=0.1):

    model.fit(X, y)
    residual = model.conformity(y, model.predict(X))
    q_alpha = np.quantile(residual, 1 - alpha)
    mu = model.predict(X[-1, :])
    lb = mu - q_alpha
    ub = mu + q_alpha

    return [lb, ub]


def splitCP(X, y, model, alpha=0.1):

    X_train, X_test, Y_train, Y_test = train_test_split(X[:-1], y,
                                                        test_size=0.5)

    model.fit(X_train, Y_train)

    # Ranking on the calibration set
    sorted_residual = np.sort(model.conformity(Y_test, model.predict(X_test)))
    index = int((X.shape[0] / 2 + 1) * (1 - alpha))

    # Double check index - 1 (because numpy tab start at 0)
    quantile = sorted_residual[index]
    mu_ = model.predict(X[-1, :])

    return [mu_ - quantile, mu_ + quantile]


def splitCPP(X, y, model, alpha=0.1):

    scp = splitCP(X, y, model, alpha)
    mu = 0.5 * (scp[0] + scp[1])
    # y_mu = np.array(list(y) + [mu[0]])
    y_mu = np.array(list(y) + [0])

    model.fit(X, y_mu)
    y_pred = model.predict(X[-1])

    print("y_pred =", y_pred, "y_scp =", mu)

    if scp[0] <= y_pred and y_pred <= scp[1]:
        bound = max(scp[1] - y_pred, y_pred - scp[0])
        print("ok", scp[1] - scp[0], bound)
        return [y_pred - bound, y_pred + bound]

    return scp


def ridgeCP(X, y, lmd, alpha=0.1):

    n_samples, n_features = X.shape
    H = X.T.dot(X) + lmd * np.eye(n_features)
    C = np.eye(n_samples) - X.dot(np.linalg.solve(H, X.T))
    A = C.dot(list(y) + [0])
    B = C[:, -1]

    negative_B = np.where(B < 0)[0]
    A[negative_B] *= -1
    B[negative_B] *= -1
    S, U, V = [], [], []

    for i in range(n_samples):

        if B[i] != B[-1]:
            tmp_u_i = (A[i] - A[-1]) / (B[-1] - B[i])
            tmp_v_i = -(A[i] + A[-1]) / (B[-1] + B[i])
            u_i, v_i = np.sort([tmp_u_i, tmp_v_i])
            U += [u_i]
            V += [v_i]

        elif B[i] != 0:
            tmp_uv = -0.5 * (A[i] + A[-1]) / B[i]
            U += [tmp_uv]
            V += [tmp_uv]

        if B[-1] > B[i]:
            S += [intervals.closed(U[i], V[i])]

        elif B[-1] < B[i]:
            intvl_u = intervals.openclosed(-np.inf, U[i])
            intvl_v = intervals.closedopen(V[i], np.inf)
            S += [intvl_u.union(intvl_v)]

        elif B[-1] == B[i] and B[i] > 0 and A[-1] < A[i]:
            S += [intervals.closedopen(U[i], np.inf)]

        elif B[-1] == B[i] and B[i] > 0 and A[-1] > A[i]:
            S += [intervals.openclosed(-np.inf, U[i])]

        elif B[-1] == B[i] and B[i] == 0 and abs(A[-1]) <= abs(A[i]):
            S += [intervals.open(-np.inf, np.inf)]

        elif B[-1] == B[i] and B[i] == 0 and abs(A[-1]) > abs(A[i]):
            S += [intervals.empty()]

        elif B[-1] == B[i] and A[-1] == A[i]:
            S += [intervals.open(-np.inf, np.inf)]

        else:
            print("boom !!!")

    hat_y = np.sort([-np.inf] + U + V + [np.inf])
    size = hat_y.shape[0]
    conf_pred = intervals.empty()
    p_values = np.zeros(size)

    for i in range(size - 1):

        n_pvalue_i = 0.
        intvl_i = intervals.closed(hat_y[i], hat_y[i + 1])

        for j in range(n_samples):
            n_pvalue_i += intvl_i in S[j]

        p_values[i] = n_pvalue_i / n_samples

        if p_values[i] > alpha:
            conf_pred = conf_pred.union(intvl_i)

    return conf_pred, hat_y, p_values


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=.75)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def load_data(dataset="boston"):

    if dataset == "boston":
        boston = datasets.load_boston()
        X_ = boston.data
        Y_ = boston.target

    if dataset == "diabetes":
        diabetes = datasets.load_diabetes()
        X_ = diabetes.data
        Y_ = diabetes.target

    if dataset == "climate":
        X_ = np.load("Xclimate.npy")
        Y_ = np.load("yclimate.npy")

        n_features = X_.shape[1]
        groups = np.arange(n_features) // 7
        groups = groups.astype(int)
        n_groups = int(n_features / 7)
        size_groups = 7 * np.ones(n_groups)
        size_groups = size_groups.astype(int)
        omega = np.ones(n_groups)  # since all groups have the same size
        g_start = np.cumsum(size_groups, dtype=np.intc) - size_groups[0]

    if dataset == "housingcalifornia":
        housing = datasets.fetch_california_housing()
        X_, Y_ = housing.data, housing.target

    if dataset == "friedman1":
        X_, Y_ = datasets.make_friedman1(
            n_samples=500, n_features=100, noise=1)

    if dataset == "synthetic":
        dense = 0.7
        n_samples, n_features = (100, 1000)
        X_, Y_ = make_regression(n_samples=n_samples, n_features=n_features,
                                 # random_state=random_state,
                                 n_informative=int(n_features * dense),
                                 noise=1)

    # without n_informativelization scipy.minimize fails to converge
    X_ /= np.linalg.norm(X_, axis=0)
    mask = np.sum(np.isnan(X_), axis=0) == 0
    if np.any(mask):
        X_ = X_[:, mask]
    Y_ = (Y_ - Y_.mean()) / Y_.std()

    return X_, Y_


def load_model(method="ridge", X=None, y=None):

    # if random_state is None, the estimator is random itself (an additional
    # randomness potentially independent of the data) and the very definition
    # of conformal set is unclear

    if method == "GradientBoosting":
        model = GradientBoostingRegressor(warm_start=True, random_state=0)

    if method == "MLP":
        model = MLPRegressor(warm_start=False, random_state=0, max_iter=2000)

    if method == "AdaBoost":
        model = AdaBoostRegressor(random_state=0)

    if method == "RandomForest":
        # For randomForest I dont know yet if it is safe to use warm_start
        model = RandomForestRegressor(warm_start=False, random_state=0)

    if method == "OMP":
        # Do not have a warm_start
        tol_omp = 1e-3 * np.linalg.norm(y) ** 2
        model = OrthogonalMatchingPursuit(tol=tol_omp, fit_intercept=False)

    if method == "Lasso":
        lmd = np.linalg.norm(X.T.dot(y), ord=np.inf) / 25
        model = Lasso(alpha=lmd / X.shape[0], warm_start=False, max_iter=5000,
                      fit_intercept=False)

    if method == "ridge":
        lmd = 0.1
        model = models.ridge(lmd=lmd)

    return models.regressor(model=model)
