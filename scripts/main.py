import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression

from model import Classifier
from metric import Metric
from inf import InfluenceFunction
from ctf import distributional_descent
from transport import Preprocessor
from util import get_data, data_split, auc, l2_cost, get_constants


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type = str, default = 'adult')
	parser.add_argument('--metric', type = str, default = 'SP')
	parser.add_argument('--seed', type = int, default = 0)

	args = parser.parse_args()

	np.random.seed(args.seed)

	x, y, s = get_data(args.dataset, args.metric)
	train_idx, ctf_idx, eval_idx = data_split(x.shape[0])

	x_train = x[train_idx]
	x_ctf = x[ctf_idx]
	x_eval = x[eval_idx]
	
	y_train = y[train_idx]
	y_ctf = y[ctf_idx]
	y_eval = y[eval_idx]
	
	s_train = s[train_idx]
	s_ctf = s[ctf_idx]
	s_eval = s[eval_idx]

	h = Classifier(x_train, y_train)
	y0_hat = Classifier(x_train[s_train == 0], y_train[s_train == 0])

	pred_ctf = h(x_ctf)

	metric = Metric(args.metric)
	psi = InfluenceFunction(args.metric, h, y0_hat, *get_constants(y_ctf, pred_ctf, s_ctf))

	weights, target_idx, target_idx_tilde = distributional_descent(x_ctf, y_ctf, s_ctf, pred_ctf, metric, psi)

	T = Preprocessor(x_ctf, weights, target_idx, l2_cost)

	x_eval_preprocessed = x_eval.copy()
	x_eval_preprocessed[s_eval == 0] = T(x_eval[s_eval == 0])

	pred_original = h(x_eval, soft = True)
	pred_repaired = h(x_eval_preprocessed, soft = True)

	print(f'Dataset: {args.dataset}')
	print('-' * 50)
	print('Original model')
	print(f'AUC: {auc(y_eval, pred_original)}')
	print(f'{args.metric}: {metric(s_eval, y_eval, np.round(pred_original))}')
	print('-' * 50)
	print('Repaired model')
	print(f'AUC: {auc(y_eval, pred_repaired)}')
	print(f'{args.metric}: {metric(s_eval, y_eval, np.round(pred_repaired))}')
	print('-' * 50)


if __name__ == '__main__':
	main()
