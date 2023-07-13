from .model import Model
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import numpy as np
import os
from .. import logger
np.random.RandomState(seed=1)

def train_model(
	data_path, 
	processed_data_path, 
	train_data_path, 
	data_name, 
	label_column, 
	model_path, 
	model_name
	):
	'''
	Функция осуществляет обучение модели на предобработанных и размеченных данных, 
	результат обучения сохраняется
	'''
	logger.logging('---START TRAINING---')
	data = pd.read_csv(os.path.join(data_path, processed_data_path, train_data_path, data_name))
	model = Model().initialize_model()
	model.fit(data.drop(label_column, axis=1), data[label_column])
	pickle.dump(model, open(os.path.join(model_path, model_name), 'wb'))
	logger.logging('---TRAINING IS COMPLETED---')