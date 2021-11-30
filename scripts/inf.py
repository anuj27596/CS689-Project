
class InfluenceFunction():
	def __init__(self, metric, h, y0_hat, mu, mu_hat, gamma):
		self.metric = metric
		self.h = h
		self.y0_hat = y0_hat
		self.mu = mu
		self.mu_hat = mu_hat
		self.gamma = gamma

	def __call__(self, x):
		if self.metric == 'SP':
			return -self.h(x) + self.mu_hat[0]

		elif self.metric == 'FNR':
			return ((1 - self.h(x)) * self.y0_hat(x) - self.gamma[0, 1] * self.y0_hat(x)) / self.mu[0]

		elif self.metric == 'FPR':
			return (self.h(x) * (1 - self.y0_hat(x)) - self.gamma[1, 0] * (1 - self.y0_hat(x))) / (1 - self.mu[0])

		else:
			return 0
