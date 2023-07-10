from .model import Model
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import numpy as np
import os
np.random.RandomState(seed=1)

def train_model(path_to_project, data_name='data.csv', label_column='label'):
	'''
	Функция осуществляет обучение модели на предобработанных и размеченных данных, 
	результат обучения сохраняется
	'''
	print('---START TRAINING---')
	data = pd.read_csv(os.path.join(path_to_project, 'data/processed/train/' + data_name))
	model = Model().initialize_model()
	model.fit(data.drop(label_column, axis=1), data[label_column])
	pickle.dump(model, open(os.path.join(path_to_project, 'models/' + 'LogRegr.pkl'), 'wb'))