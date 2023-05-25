import numpy as np
from sklearn.model_selection import train_test_split
from scipy import optimize
from scipy.special import expit


def rank(u, gamma=None):

    if gamma is None:
        return np.sum(u - u[-1] <= 0)

    sigmoid = 1 - expit(gamma * (u - u[-1]))
    return np.sum(sigmoid)


def conformalset(X, y, model, alpha=0.1, gamma=None, nqp=4, tol=1e-3):

    # Localization
    X_train, X_test, Y_train, Y_test = train_test_split(X[:-1, :], y,
                                                        test_size=0.33)

    model.fit(X_train, Y_train)
    sorted_residual = np.sort(model.conformity(Y_test, model.predict(X_test)))
    index = int((len(Y_test) + 1) * (1 - alpha))
    quantile = sorted_residual[index]
    pred_ = model.predict(X[-1, :])
    scp = [pred_ - quantile, pred_ + quantile]

    # i_lb = scp[0] - 0.25 * (scp[1] - scp[0])
    # i_ub = scp[1] + 0.25 * (scp[1] - scp[0])

    i_lb = scp[0] - 0.5 * (scp[1] - scp[0])
    i_ub = scp[1] + 0.5 * (scp[1] - scp[0])

    left_query_point = np.linspace(i_lb, pred_, nqp)
    right_query_point = np.linspace(pred_, i_ub, nqp)
    xs = np.unique(list(left_query_point) + list(right_query_point))
    yz = np.asfortranarray(list(y) + [0])

    # Interpolation of model fit
    def interpol_model_fit(zs):

        preds = []

        for z in zs:
            yz[-1] = z
            model.fit(X, yz)
            preds += [model.predict(X)]

        return preds

    preds = interpol_model_fit(xs)

    def interpolated_model_fit(X, obs, zs=xs, preds=preds):

        z = obs[-1]

        if z < zs[0]:
            iz = 0
            sz = (zs[iz] - z) / (zs[iz + 1] - zs[iz])
            return (sz + 1) * preds[iz] - sz * preds[iz + 1]

        if z > zs[-1]:
            iz = -1
            sz = (z - zs[iz]) / (zs[iz] - zs[iz - 1])
            return (sz + 1) * preds[iz] - sz * preds[iz - 1]

        iz = np.where(zs <= z)[0][-1]

        if iz == len(zs) - 1:
            return preds[-1]

        sz = (z - zs[iz + 1]) / (zs[iz] - zs[iz + 1])

        return sz * preds[iz] + (1 - sz) * preds[iz + 1]

    def pvalue(z):

        yz[-1] = z
        pred = interpolated_model_fit(X, yz, zs=xs, preds=preds)
        scores = model.conformity(yz, pred)

        return 1 - rank(scores, gamma) / X.shape[0]

    def objective(z):
        # we use a bit more conservative alpha to include all roots when
        # the p_value function is piecewise constant.
        return pvalue(z) - (alpha - 1e-11)

    objs = [objective(z) for z in xs]
    z_0 = xs[np.argmax(objs)]

    y_min = np.min(y)
    y_max = np.max(y)
    # z_min = y_min - 0.5 * (y_max - y_min)
    # z_max = y_max + 0.5 * (y_max - y_min)
    z_min = y_min - 2 * (y_max - y_min)
    z_max = y_max + 2 * (y_max - y_min)
    z_min = min(i_lb, z_min)
    z_max = max(i_ub, z_max)
    # TODO: cleaner way to handle this

    if objective(z_0) < 0 or objective(z_min) > 0 or objective(z_max) > 0:
        print("interpol init fail")
        return scp

    left = optimize.root_scalar(objective, bracket=[z_min, z_0],
                                method='brenth', xtol=tol)

    right = optimize.root_scalar(objective, bracket=[z_0, z_max],
                                 method='brenth', xtol=tol)

    return [left.root, right.root]
