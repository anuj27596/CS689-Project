import numpy as np
import pandas as pd


def get_data(dataset, metric):
	data_path, sensitive_values = None, None

	if dataset == 'adult':
		data_path = '../data/adult_proxy_data.csv'
		sensitive_values = ('Male', 'Female') if metric in ['FPR'] else ('Female', 'Male')
	elif dataset == 'compas':
		data_path = '../data/compas_proxy_data.csv'
		sensitive_values = ('non-white', 'white') if metric in ['FPR'] else ('white', 'non-white')

	df = pd.read_csv(data_path)
	N = len(df)

	data = np.array(df)

	x = data[:, 2:]
	
	y = data[:, 0]
	y[y == -1] = 0
	
	s = data[:, 1]
	s[s == sensitive_values[0]] = 0
	s[s == sensitive_values[1]] = 1

	return x.astype(int), y.astype(int), s.astype(int)


def data_split(n, train_fraction = 0.3, ctf_fraction = 0.5, eval_fraction = 0.2):
	idx = np.arange(n)
	np.random.shuffle(idx)

	train_idx = idx[:round(train_fraction * n)]
	ctf_idx = idx[round(train_fraction * n):round((train_fraction + ctf_fraction) * n)]
	eval_idx = idx[-round(eval_fraction * n):]

	return train_idx, ctf_idx, eval_idx


def auc(y, pred):
	pos = tp = y.sum()
	neg = fp = y.shape[0] - pos

	auc = 0

	for i in pred.argsort()[:-1]:
		if y[i] == 1:
			tp -= 1
		else:
			fp -= 1
			auc += tp / (pos * neg)

	return auc


def get_constants(y, pred, s):
	mu = np.array([y[s == x].mean() for x in [0, 1]])
	mu_hat = np.array([pred[s == x].mean() for x in [0, 1]])
	gamma = np.array([[np.abs(pred[(y == b) & (s == 0)] + a - 1).mean() for b in [0, 1]] for a in [0, 1]])

	return mu, mu_hat, gamma


def encode(x):
	return x @ (2 ** np.arange(x.shape[-1]))


def decode(e, n):
	return e[:, None] // (2 ** np.arange(n)) % 2


def l2_cost(x, y):
	return bin(x ^ y).count('1') ** 2


def sigmoid(x):
	return 1 / (1 + np.e ** -x)
