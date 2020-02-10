"""
	DBSCAN clustering
	Also, implementation of a greedy seach of optimal hyperparameters of DBSCAN for face clustering.

	author: Ricardo Kleinlein
	date: 02/2020

	Usage:
		python grid_search.py <program-csv>

	Options:
		--output-dir	Directory to save results in
		--quiet	Hide visual information
		-h, --help	Display script additional help
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from arguments import DbscanArgs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def load_embeddings(db_path, min_size=0):
	"""Load a list of vector embeddings from a
	csv file summarizing the face detection of a program

	Args:
		db_path (str): Path to csv detection file
		min_size (int, optional): Min area of face to be loaded

	Return:
		a list of vector embeddings in np.ndarray form
		a list of bouding boxes sizes (np.ndarray)
	"""
	db = pd.read_csv(db_path, usecols=
		['size', 'embedding'])
	size = db['size'].values
	emb = db['embedding'].values
	if min_size != 0:
		idx2keep = []
		for i, s in enumerate(size):
			if s >= min_size:
				idx2keep.append(i)
		emb = emb[idx2keep]
	emb = [np.load(i) for i in emb]
	return emb, size


def hist_face_sizes(X, output_dir):
	"""Save a histogram depicting the face sizes.

	Args:
		X (float): List of face sizes
		output_dir (str): Directory to save in
	"""
	os.makedirs(output_dir, exist_ok=True)
	plt.hist(X, bins=100)
	plt.xlabel('Face bounding box area [px^2]')
	plt.ylabel('Frequency')
	plt.savefig(join(output_dir, 'face_sizes.png'))


def dbscan_(X, eps, min_samples, metric='euclidean'):
	"""DBSCAN clustering for a set of parameters over the 
	sampels X.

	Args:
		X (float): Feature sample vectors
		eps (float): Epsilon hparam
		min_samples (int): Min-samples hparam
		metric (str, optional): distance metric [default: euclidean]

	Return:
		np.ndarray of labels for each samples, with 
		noisy samples given a `-1`
	"""
	f = DBSCAN(eps=eps, 
		min_samples=min_samples,
		metric=metric)
	f.fit(X)
	return f.labels_


def export(path, data):
	"""Exports data to a csv file."""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	np.savetxt(path, data)


def nearestneighbors(X, n, metric='euclidean'):
	"""Compute the distance to the n-th neighbour in an array
	of feature samples X.
	
	Args:
		X (float): Array of feature samples.
		n (int): N-th neighbor to consider.
		metric (str): DIstance measure [default: euclidean]

	Return:
		An np.ndarray of distances up to the n-th neighbors
	"""
	nn = NearestNeighbors(n_neighbors=n,
		metric=metric,
		n_jobs=-1)
	nbrs = nn.fit(X)
	dist, _ = nbrs.kneighbors(X)
	sort_dist = np.sort(dist, axis=0)[:, 1:]
	return sort_dist


def gaussian_fit(x, output_dir=None):
	"""Get stats of a vector of distances and 
	model it as a gaussian distribution.

	Args:
		x (float): Vector of distances between neighbors
		output_dir (str, optional): Saving dirname. 

	Return:
		Inferior and Upper limits for search as twice the std
	"""
	mu = np.mean(x)
	std = np.std(x)
	inf = mu + std
	up = mu + std + std
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
		plt.hist(x, bins=50)
		plt.axvline(x=mu, color='r')
		plt.axvspan(mu-std, mu+std, facecolor='y', alpha=0.2)
		plt.axvspan(inf, up, facecolor='g', alpha=0.6)
		plt.xlabel('Distance to neighbor', fontsize=12)
		plt.ylabel('Frequency', fontsize=12)
		plt.savefig(join(output_dir, 'gaussian_fit_dist_neighbor.png'))
		plt.clf()
		plt.plot(x)
		plt.axhline(y=mu, color='r')
		plt.axhspan(mu-std, mu+std, facecolor='y', alpha=0.2)
		plt.axhspan(inf, up, facecolor='g', alpha=0.6)
		plt.xlabel('Sample number', fontsize=12)
		plt.ylabel('Distance', fontsize=12)
		plt.savefig(join(output_dir, 'dist_neighbor.png'))
	return (inf, up)


def get_n_noise_samples(labels):
	return list(labels).count(-1)


def get_number_clusters(labels):
	return len(set(labels)) - (1 if -1 in labels else 0)


if __name__ == "__main__":
	args = DbscanArgs().parse()
	X, sizes = load_embeddings(args.program_csv,
		args.min_face_size)
	if not args.eps and not args.min_samples:
		if not args.quiet:
			print('> Proceed to hyperparameters search...')
		dirname = join(args.output_dir, 'dbscan_hparams_' + args.metric)
		filename = 'dist_' + str(args.nthneigh - 1) + 'th_neighbor.csv'

		# dists = nearestneighbors(X, args.nthneigh, metric=args.metric)
		# export(join(dirname, filename), data=dists)
		hist_face_sizes(sizes, args.output_dir)
		# gaussian_fit(dists[:, args.nthneigh-2], dirname)


		if not args.quiet:
			print('> Distance to neighbors exported to file')
	elif args.eps and args.min_samples:
		labels_ = dbscan_(X=X,
			eps=args.eps,
			min_samples=args.min_samples,
			metric=args.metric)
	else:
		raise IOError("Missing one hyperparameter")

