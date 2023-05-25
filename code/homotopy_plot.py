import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import matplotlib
from rootCP.rootcp.tools import set_style
# plt.rcParams["text.usetex"] = True
set_style()
matplotlib.rcParams.update({'font.size': 85})
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_path(filename, losstype):
	with open(filename, "rb") as f:
		dbetas, dkinks, betas, kinks = pickle.load(f)
	dbetas = np.asarray(dbetas)
	betas = np.asarray(betas)
	stopping_idx = -1
	cmap = get_cmap(dbetas.shape[1])
	plt.clf()
	for i in range(dbetas.shape[1]):
		if i == dbetas.shape[1] - 1:
			plt.plot(dkinks, dbetas[:, i], '-', color=cmap(i), label="Grid")
			plt.plot(kinks, betas[:, i], '--', color=cmap(i), label="Ours")
		else:
			plt.plot(dkinks, dbetas[:, i], '-', color=cmap(i))
			plt.plot(kinks, betas[:, i], '--', color=cmap(i))
		# plt.plot(, label="O{}".format(i))
	plt.xlabel(r'$y_{n+1}$')
	plt.ylabel(r'$\beta_j$')

	plt.legend()
	plt.tight_layout()
	plt.grid(None)
	plt.savefig("code/official_images/path_{}.png".format(losstype),  bbox_inches='tight',  dpi=300)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--filename")
	parser.add_argument("--losstype")

	args = parser.parse_args()
	plot_path(args.filename, args.losstype)