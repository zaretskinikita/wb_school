'''
В засисимости от MODE (Train(0), Inference(1) или Default(2) обрабатывает сырые данные и загружает данные, пригодные для обучения или применения модели в папку data/processed.
# 	В Default mode программа ищет данные и для обучения и для применения
'''
import warnings
warnings.filterwarnings('ignore')
# from params import Parameters
from .preprocessing import *
import pandas as pd
import os
import pickle
# import numpy as np
import argparse
from sklearn.model_selection import cross_val_score
np.random.RandomState(seed=1)

def make_processed_data(path_to_project, data_name='data.csv', mode=2, label_column='label', text_column='text'):
	'''
	Функция осуществляет предобработку данных в засисимости от целей (Train/Inference/Default)
	'''
	print('---START PREPROCESSING---')
	#1. Загрузка данных 
	if mode == 0 or mode == 2:
		# TRAIN OR DEFAULT
		if mode == 0:
			print('\t---TRAIN MODE---')
		else:
			print('\t---DEFAULT MODE---')

		file_name1 = 'train/' + data_name
		try:
			data = pd.read_csv(os.path.join(path_to_project, 'data/raw/' + file_name1), lineterminator='\n')
			if data.shape[1] != 13:
				raise KeyError('Train data must contain 13 columns')
			data = Preprocessing(data, cols_to_remove=['id1', 'id2', 'id3'], text_col=text_column, label_col=label_column).full_preprocessing()
			data.to_csv(os.path.join(path_to_project, 'data/processed/' + file_name1), index=False)
		except FileNotFoundError:
			print('\t\t---FILE ' + os.path.join(path_to_project, 'data/raw/' + file_name1) + ' IS NOT FOUND---')
		print('\t---TRAIN PREPROCESSING IS FINISHED')
	if mode == 1 or mode == 2:
		# INFERENCE OR DEFAULT
		if mode == 1:
			print('\t---INFERENCE MODE---')
		file_name2 = 'inference/' + data_name
		try:
			data = pd.read_csv(os.path.join(path_to_project, 'data/raw/' + file_name2), lineterminator='\n')
			if data.shape[1] != 12:
				raise KeyError('Inference data must contain 12 columns')
			data = Preprocessing(data, cols_to_remove=['id1', 'id2', 'id3'], text_col=text_column, label_col='').full_preprocessing()
			data.to_csv(os.path.join(path_to_project, 'data/processed/' + file_name2), index=False)
		except FileNotFoundError:
			print('\t\t---FILE ' + os.path.join(path_to_project, 'data/raw/' + file_name1) + ' IS NOT FOUND---')
		print('\t---INFERENCE PREPROCESSING IS FINISHED')
	print('---DATA PREPROCESSING IS COMPLETED---')