import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib
import argparse
from rootCP.rootcp.tools import set_style
# plt.rcParams["text.usetex"] = True
set_style()
matplotlib.rcParams.update({'font.size': 85})
def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	return plt.cm.get_cmap(name, n)

def plot_path(filename, losstype, dataset):
	with open(filename, "rb") as f:
		d_ranks, search_space, our_ranks, our_points = pickle.load(f)
	norm_constant = max(max(d_ranks), max(our_ranks))
	d_ranks = np.asarray(d_ranks)/norm_constant
	print("hi")
	our_ranks = np.asarray(our_ranks)/norm_constant
	plt.plot(search_space,d_ranks, label="Grid")
	plt.step(our_points, our_ranks, label="Ours", where="post")
	plt.legend(fontsize=15)
	plt.grid(None)
	plt.xlabel(r'$y_{n+1}$', fontsize=15)
	plt.ylabel(r'$\pi(z)$', fontsize=15)
	plt.tight_layout()
	# plt.rc('axes', titlesize=100)     # fontsize of the axes title
	# plt.rc('axes', labelsize=100)    # fontsize of the x and y labels
	# plt.rc('legend', fontsize=22)  
	# params = {'axes.labelsize': 100,'axes.titlesize':100, 'legend.fontsize':100, 'xtick.labelsize':100, 'ytick.labelsize': 100}
	# matplotlib.rcParams.update(params)	
	plt.savefig("code/official_images/cpranks_{}_{}.png".format(losstype, dataset),  bbox_inches='tight',  dpi=300)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--filename")
	parser.add_argument("--losstype")
	parser.add_argument("--dataset")

	print("hi")
	args = parser.parse_args()
	plot_path(args.filename, args.losstype, args.dataset)