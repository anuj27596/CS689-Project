import numpy as np


def distributional_descent(x, y, s, pred, metric, psi, epsilon = 1e-4):
	# init
	idx_tilde = np.arange(x.shape[0])
	target_idx, = np.where(s == 0)
	n_target = target_idx.size
	w = np.ones(x.shape[0])

	M = metric(s, y, pred)
	M_old = np.inf

	f = psi(x[target_idx])
	f /= np.var(f)
	factor = 1 - epsilon * f

	# descent
	while abs(M) < abs(M_old):
		M_old = M
		w[target_idx] *= factor
		M = metric(s, y, pred, w = w)

	w[target_idx] /= factor

	idx_tilde[target_idx] = np.random.choice(target_idx, size = n_target, p = w[target_idx] / w[target_idx].sum())

	return w, target_idx, idx_tilde[target_idx]
