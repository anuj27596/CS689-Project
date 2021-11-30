import numpy as np


class Metric():
	def __init__(self, metric):
		self.metric = metric

	def __call__(self, s, y, y_hat, w = None):

		if w is None:
			w = np.ones_like(s)

		y_hat = np.round(y_hat)

		if self.metric == 'SP':
			s0 = s == 0
			s1 = s == 1
			return ((1 - y_hat) * w)[s0].sum() / w[s0].sum() - ((1 - y_hat) * w)[s1].sum() / w[s1].sum()

		elif self.metric == 'FNR':
			s0 = (y == 1) & (s == 0)
			s1 = (y == 1) & (s == 1)
			return ((1 - y_hat) * w)[s0].sum() / w[s0].sum() - ((1 - y_hat) * w)[s1].sum() / w[s1].sum()

		elif self.metric == 'FPR':
			s0 = (y == 0) & (s == 0)
			s1 = (y == 0) & (s == 1)
			return (y_hat * w)[s0].sum() / w[s0].sum() - (y_hat * w)[s1].sum() / w[s1].sum()

		else:
			return 0
