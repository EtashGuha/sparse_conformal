import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
from rootCP.rootcp.tools import set_style
# plt.rcParams["text.usetex"] = True
set_style()

def generate_images_gap_eps(filename, loss_type, dataset):
	with open(filename, "rb") as f:
		print(picle.load(f))
		our_gaps, our_primal_gaps, search_space = pickle.load(f)

	plt.clf()
	plt.plot(search_space, our_primal_gaps, label="Primal Gap")
	plt.xlabel(r'$y_{n+1}$')
	plt.ylabel("Objective Error")

	plt.yscale("log")

	plt.savefig("code/official_images/eps_{}_{}.png".format(loss_type, dataset),  bbox_inches='tight',  dpi=300)

def generate_images_homotopy(filename, loss_type):
	with open(filename, "rb") as f:
		our_results, approx_results = pickle.load(f)

	dims = list(our_results.keys())
	result_dicts = [our_results, approx_results]
	all_results = []
	for result_dict in result_dicts:
		gaps = []
		times = []
		num_kinks = []
		for dim in dims:
			
			try:
				curr_gaps, curr_times, curr_num_kinks = result_dict[dim]
				
				gaps.append(np.mean(curr_gaps))
				times.append(np.mean(curr_times))
				num_kinks.append(np.mean(curr_num_kinks))
			except:
				gaps.append(0)
				times.append(0)
				num_kinks.append(0)
		all_results.append((gaps, times, num_kinks))

	names = ["gap", "time", "numkinks"]
	axes = ["Average Negative Log of the Duality Gap", "Average Time (s)", "Number of Active Set Changes"]
	methods = ["Ours", "Approximate Homotopy"]
	for metric in range(len(axes)):
		plt.clf()
		for method in range(len(methods)):
			plt.plot(dims, all_results[method][metric], label=methods[method])
		plt.legend()
		plt.xlabel("Dimension")
		plt.ylabel(axes[metric])
		plt.savefig("code/official_images/homotopy_{}_{}.png".format(loss_type, names[metric]),  bbox_inches='tight',  dpi=300)

def generate_images_conformal(filename, loss_type):
	with open(filename, "rb") as f:
		breakpoint()
		our_results, approx_results, root_results, split_results  = pickle.load(f)

	dims = list(our_results.keys())
	result_dicts = [our_results, approx_results, root_results, split_results]
	all_results = []
	for result_dict in result_dicts:
		coverages = []
		lengths = []
		times = []
		for dim in dims:
			try:
				curr_coverages, curr_lengths, curr_times = result_dict[dim]
				
				coverages.append(np.mean(curr_coverages))
				lengths.append(np.mean(curr_lengths))
				times.append(np.mean(curr_times))
			except:
				coverages.append(0)
				lengths.append(0)
				times.append(0)
		all_results.append((coverages, lengths, times))

	names = ["coverage", "length", "time"]
	axes = ["Coverage", "Length of Confidence Interval", "Time (s)"]
	methods = ["Ours", "Approximate Homotopy", "RootCP", "Split"]
	# methods = ["Ours",  "Approximate Homotopy", "RootCP",]
	for metric in range(len(axes)):
		plt.clf()
		for method in range(len(methods)):
			plt.plot(dims, all_results[method][metric], label=methods[method])
		plt.legend()
		plt.xlabel("Dimension")
		plt.ylabel(axes[metric])
		
		plt.savefig("code/official_images/conformal_{}_{}.png".format(loss_type, names[metric]),  bbox_inches='tight',  dpi=300)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--loss-type", choices=["assymetric", "lasso", "robust"])
	parser.add_argument("--dataset")

	parser.add_argument("--file-name")
	
	args = parser.parse_args()
	# generate_images_homotopy(args.file_name, args.loss_type)
	generate_images_conformal(args.file_name, args.loss_type)
	# generate_images_gap_eps(args.file_name, args.loss_type, args.dataset)

	
