from .model import Model
from .params import Parameters
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_score
import pickle
import os
import numpy as np
np.random.RandomState(seed=1)

def predict_model(path_to_project, data_name='data.csv', model_name='LogRegr.pkl'):
	'''
	Функция осуществляет применение модели (сохраненной) к предобработанным данным,
	результат сохраняется
	'''
	print('---START INFERENCE---')
	data = pd.read_csv(os.path.join(path_to_project, 'data/processed/inference/' + data_name))
	model = pickle.load(open(os.path.join(path_to_project, 'models/' + model_name), 'rb'))
	preds = model.predict_proba(data)[:, 1]
	preds[preds < Parameters.threshold] = 0
	preds[preds > Parameters.threshold] = 1
	pickle.dump(preds, open(os.path.join(path_to_project, 'models/' + 'LogRegrPredictions.pkl'), 'wb'))