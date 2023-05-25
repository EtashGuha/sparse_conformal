import pandas as pd 
import pickle
import numpy as np
datasets = ["diabetes"]

losses = ["lasso", "robust", "assymetric"]
coverage_dict = {}
length_dict = {}
time_dict = {}

for dataset in datasets:
    for loss in losses:
        name = "/nethome/eguha3/predictor_corrector_cp/results/cp_{}_{}_0.1a_results.pkl".format(dataset, loss)
        our_results, approx_results, root_results, split_results, grid_results, oracle_results = pickle.load(open(name, "rb"))
        coverage_dict[loss] = dict()
        coverage_dict[loss]["Ours"] = np.mean(our_results["ours"][0])
        coverage_dict[loss]["Approximate"] = np.mean(approx_results["approx"][0])
        coverage_dict[loss]["Split"] = np.mean(split_results["split"][0])
        coverage_dict[loss]["Grid"] = np.mean(grid_results["grid"][0])
        coverage_dict[loss]["Oracle"] = np.mean(oracle_results["oracle"][0])

        length_dict[loss] = dict()
        length_dict[loss]["Ours"] = np.mean(np.asarray(our_results["ours"][1])[np.nonzero(our_results["ours"][1])])
        length_dict[loss]["Approximate"] = np.mean(np.asarray(approx_results["approx"][1])[np.nonzero(approx_results["approx"][1])])
        length_dict[loss]["Split"] = np.mean(np.asarray(split_results["split"][1])[np.nonzero(split_results["split"][1])])
        length_dict[loss]["Grid"] = np.mean(np.asarray(grid_results["grid"][1])[np.nonzero(grid_results["grid"][1])])
        length_dict[loss]["Oracle"] = np.mean(np.asarray(oracle_results["oracle"][1])[np.nonzero(oracle_results["oracle"][1])])

        time_dict[loss] = dict()
        time_dict[loss]["Ours"] = np.mean(np.asarray(our_results["ours"][2])[np.nonzero(our_results["ours"][2])])
        time_dict[loss]["Approximate"] = np.mean(np.asarray(approx_results["approx"][2])[np.nonzero(approx_results["approx"][2])])
        time_dict[loss]["Split"] = np.mean(np.asarray(split_results["split"][2])[np.nonzero(split_results["split"][2])])
        time_dict[loss]["Grid"] = np.mean(np.asarray(grid_results["grid"][2])[np.nonzero(grid_results["grid"][2])])
        time_dict[loss]["Oracle"] = np.mean(np.asarray(oracle_results["oracle"][2])[np.nonzero(oracle_results["oracle"][2])])

    res = pd.DataFrame.from_dict(coverage_dict)
    res.to_csv("results/{}_coverage.csv".format(dataset))

    res = pd.DataFrame.from_dict(length_dict)
    res.to_csv("results/{}_length.csv".format(dataset))

    res = pd.DataFrame.from_dict(time_dict)
    res.to_csv("results/{}_time.csv".format(dataset))
