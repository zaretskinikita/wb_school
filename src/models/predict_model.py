from .model import Model
from .. import config
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_score
import pickle
import os
import numpy as np
from .. import logger
np.random.RandomState(seed=1)

def predict_model(
	data_path, 
	processed_data_path,
	inference_data_path,
	data_name,
	model_path, 
	model_name,
	preds_name
	):
	'''
	Функция осуществляет применение модели (сохраненной) к предобработанным данным,
	результат сохраняется
	'''
	logger.logging('---START INFERENCE---')
	data = pd.read_csv(os.path.join(data_path, processed_data_path, inference_data_path, data_name))
	model = pickle.load(open(os.path.join(model_path, model_name), 'rb'))
	preds = model.predict_proba(data)[:, 1]
	preds[preds < config.Parameters.threshold] = 0
	preds[preds > config.Parameters.threshold] = 1
	pickle.dump(preds, open(os.path.join(model_path, preds_name), 'wb'))
	logger.logging('---INFERENCE IS COMPLETED---')