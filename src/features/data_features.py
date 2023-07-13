'''
В засисимости от MODE (Train(0), Inference(1) или Default(2) обрабатывает сырые данные и загружает данные, пригодные для обучения или применения модели в папку data/processed.
# 	В Default mode программа ищет данные и для обучения и для применения
'''
import warnings
warnings.filterwarnings('ignore')
from .preprocessing import *
import pandas as pd
import os
import pickle
import argparse
from sklearn.model_selection import cross_val_score
np.random.RandomState(seed=1)
from .. import logger

def make_processed_data(
	data_path, 
	raw_data_path, 
	processed_data_path, 
	train_data_path, 
	inference_data_path,
	data_name, 
	mode, 
	label_column, 
	text_column, 
	cols_to_remove
	):
	
	'''
	Функция осуществляет предобработку данных в засисимости от целей (Train/Inference/Default).
	'''

	logger.logging('---START PREPROCESSING---')
	#1. Загрузка данных 
	if mode == 0 or mode == 2:
		# TRAIN OR DEFAULT
		if mode == 0:
			logger.logging('\t---TRAIN MODE---')
		else:
			logger.logging('\t---DEFAULT MODE---')

		# file_name1 = 'train/' + data_name
		try:
			data = pd.read_csv(os.path.join(data_path, raw_data_path, train_data_path, data_name), lineterminator='\n')
			if data.shape[1] != 13:
				raise KeyError('Train data must contain 13 columns')
			data = Preprocessing(data, cols_to_remove=cols_to_remove, text_col=text_column, label_col=label_column).full_preprocessing()
			data.to_csv(os.path.join(data_path, processed_data_path, train_data_path, data_name), index=False)
		except FileNotFoundError:
			logger.logging('\t\t---FILE ' + os.path.join(data_path, raw_data_path, train_data_path, data_name) + ' IS NOT FOUND---')
		logger.logging('\t---TRAIN PREPROCESSING IS FINISHED')
	if mode == 1 or mode == 2:
		# INFERENCE OR DEFAULT
		if mode == 1:
			logger.logging('\t---INFERENCE MODE---')
		# file_name2 = 'inference/' + data_name
		try:
			data = pd.read_csv(os.path.join(data_path, raw_data_path, inference_data_path, data_name), lineterminator='\n')
			if data.shape[1] != 12:
				raise KeyError('Inference data must contain 12 columns')
			data = Preprocessing(data, cols_to_remove=cols_to_remove, text_col=text_column, label_col='').full_preprocessing()
			data.to_csv(os.path.join(data_path, processed_data_path, inference_data_path, data_name), index=False)
		except FileNotFoundError:
			logger.logging('\t\t---FILE ' + os.path.join(data_path, raw_data_path, inference_data_path, data_name) + ' IS NOT FOUND---')
		logger.logging('\t---INFERENCE PREPROCESSING IS FINISHED')
	logger.logging('---DATA PREPROCESSING IS COMPLETED---')
