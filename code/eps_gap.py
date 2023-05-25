import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import matplotlib
from rootCP.rootcp.tools import set_style
# plt.rcParams["text.usetex"] = True
set_style()
matplotlib.rcParams.update({'font.size': 85})

def generate_images_gap_eps(lassofilename, robustfilename, assymetricfilename, datasetname):
	with open(lassofilename, "rb") as f:
		lasso_our_gaps, lasso_our_primal_gaps, lasso_search_space = pickle.load(f)
	with open(robustfilename, "rb") as f:
		robust_our_gaps, robust_our_primal_gaps, robust_search_space = pickle.load(f)
	with open(assymetricfilename, "rb") as f:
		assymetric_our_gaps, assymetric_our_primal_gaps, assymetric_search_space = pickle.load(f)
	plt.clf()
	plt.plot(lasso_search_space, lasso_our_primal_gaps, label="Lasso")
	plt.plot(robust_search_space, robust_our_primal_gaps, label="Robust")
	plt.plot(assymetric_search_space, assymetric_our_primal_gaps, label="Asymetric")
	plt.legend()
	plt.grid(None)
	plt.tight_layout()


	plt.xlabel(r'$y_{n+1}$')
	plt.ylabel("Objective Error")

	plt.yscale("log")

	plt.savefig("code/official_images/eps_{}.png".format(datasetname),  bbox_inches='tight',  dpi=300)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--lassodataset")
	parser.add_argument("--robustdataset")
	parser.add_argument("--assymetricdataset")
	parser.add_argument("--datasetname")



	parser.add_argument("--file-name")
	
	args = parser.parse_args()
	# generate_images_homotopy(args.file_name, args.loss_type)
	# generate_images_conformal(args.file_name, args.loss_type)
	generate_images_gap_eps(args.lassodataset, args.robustdataset, args.assymetricdataset, args.datasetname)

	
