import numpy as np
import intervals
from rootcp.tools import oracleCP, splitCP, set_style, load_data, load_model
import time
from matplotlib import pyplot as plt
from rootcp import rootCP, interpolationCP
import pandas as pd

plt.rcParams["text.usetex"] = True
set_style()

random_state = np.random.randint(100)
np.random.RandomState(random_state)
print("random_state = ", random_state)

run_bench = True
save_bench = True

run_plot_bench = True
show_bench = False
save_figure = True


# The results will be averaged over n_repet randomized repetitions
n_repet = 10

# Coverage levels
alpha = 0.1
# alphas = np.arange(1, 10) / 10


def run(method, dataset, CPs, n_repet=10, alpha=0.1):

    X_, Y_ = load_data(dataset)
    regression_model = load_model(method, X_, Y_)
    print("Benchmarks:", method, dataset, X_.shape)
    print("alpha =", alpha, "and n_repet =", n_repet)

    def record(cp_method, X, y_seen, y_notseen, alpha=alpha):

        tic = time.time()
        lb, ub = cp_method(X, y_seen, regression_model, alpha=alpha)
        tac = time.time() - tic
        cp_set = intervals.closed(lb, ub)

        return np.array([y_notseen in cp_set, ub - lb, tac], dtype=object)

    columns = ["coverage", "length", "time"]

    res = {}
    for key in CPs.keys():

        res[key] = pd.DataFrame(np.zeros((n_repet, len(columns))),
                                columns=columns)

    cov_range = 0.
    random_int = np.arange(Y_.shape[0])

    print("i_repet is")

    for i_repet in range(n_repet):

        print(i_repet, sep=' ', end=' ', flush=True)

        np.random.shuffle(random_int)
        X, Y = X_[random_int, :], Y_[random_int]
        y_seen, Y_left = Y[:-1], Y[-1]

        print("y_left =", Y_left)

        # Range
        Y_range = np.min(y_seen), np.max(y_seen)
        cov_range += Y_left in intervals.closed(Y_range[0], Y_range[1])

        for key in res.keys():

            obs = Y if key == "oracleCP" else y_seen
            res[key].iloc[i_repet] = record(CPs[key], X, obs, Y_left, alpha)

    if save_bench:
        np.save(method + "_" + dataset + ".npy", res)

    # return res


def plot(method, dataset, CPs, colors):

    print("Plot:", method, dataset)

    def labelize(name, df, norm=1):

        cov = r"$\overline{cov}$ = "
        # mean_cov = str(df["coverage"].mean())
        mean_cov = str(np.round(df["coverage"].mean(), 2))
        Ts = r"$\overline{T}$ = "
        mean_time = str(np.round(df["time"].mean() / norm, 2))

        return name + " \n" + cov + mean_cov + "\n" + Ts + mean_time
        # return name + " \n" + cov + mean_cov

    name = method + "_" + dataset
    res = np.load(name + ".npy", allow_pickle=True).tolist()

    labels = []
    df_length = []
    for key in res.keys():
        print("\n ", key, "\n", res[key].mean(), "\n")
        labels += [labelize(key, res[key], res["oracleCP"]["time"].mean())]
        df_length += [res[key]["length"]]

    plt.figure()
    box = plt.boxplot(df_length, patch_artist=True)
    plt.ylabel("Length")
    plt.title(method + " : " + dataset)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.grid(False)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.tight_layout()
    name = method + "_" + dataset + ".pdf"

    if save_figure:
        plt.savefig(name, format="pdf")

    if show_bench:
        plt.show()


CPs = {"oracleCP": oracleCP,
       "splitCP": splitCP,
       "rootCP": rootCP.conformalset,
       "interpolCP": interpolationCP.conformalset,
       }

CP_colors = ["#0E0E0E", "#117733", "#882255", "#DDCC77", "k", "r"]
# datasets = ["boston", "diabetes", "housingcalifornia", "friedman1"]
datasets = ["diabetes"]
methods = ["GLM_lasso"]


if run_bench:
    [run(method, dataset, CPs, n_repet, alpha)
     for method in methods for dataset in datasets]

if run_plot_bench:
    [plot(method, dataset, CPs, CP_colors)
     for method in methods for dataset in datasets]
