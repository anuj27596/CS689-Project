import numpy as np
from pulp import LpVariable, LpProblem, LpMinimize, value, PULP_CBC_CMD

from util import encode, decode

class Preprocessor():
	def __init__(self, x, w, idx, cost):
		self.cost = cost

		enc = encode(x)
		self.D0 = list(set(enc[idx]))

		gamma = {i: {j: LpVariable(f'g_{i}_{j}', lowBound = 0) for j in self.D0} for i in self.D0}

		p, q = {}, {}

		for i in idx:
			p[enc[i]] = p.get(enc[i], 0) + 1
			q[enc[i]] = q.get(enc[i], 0) + w[i]

		p_sum = sum(p.values())
		q_sum = sum(q.values())

		for i in self.D0:
			p[i] /= p_sum
			q[i] /= q_sum


		prob = LpProblem('optimal_transport', LpMinimize)


		obj = 0

		for i in gamma.keys():
			for j in gamma[i].keys():
				obj += cost(i, j) * gamma[i][j]

		prob += obj


		for i in self.D0:
			p_lhs, q_lhs = 0, 0

			for j in self.D0:
				p_lhs += gamma[i][j]
				q_lhs += gamma[j][i]

			prob += p_lhs == p[i]
			prob += q_lhs == q[i]


		prob.solve(PULP_CBC_CMD(msg = 0, gapRel = 1e-16))

		self.mapping = {i: {j: max(0, gamma[i][j].varValue / p[i]) for j in self.D0} for i in self.D0}


	def __call__(self, x):
		enc = encode(x)

		for i, e in enumerate(enc):

			if e in self.mapping:

				p = np.array(list(self.mapping[e].values()))
				p /= p.sum()
				enc[i] = np.random.choice(self.D0, p = p)

			else:

				pass


		return decode(enc, x.shape[-1])

