from sklearn.linear_model import LogisticRegression

from util import sigmoid


class Classifier():
	def __init__(self, x, y):
		self.clf = LogisticRegression(random_state = 0, verbose = 0)
		self.clf.fit(x, y)

	def __call__(self, x, soft = True):
		if soft:
			return sigmoid(self.clf.decision_function(x))
		else:
			return (self.clf.decision_function(x) > 0).astype(int)
