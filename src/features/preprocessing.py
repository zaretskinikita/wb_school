# приведение входных данных к виду: числовые данные, попарные деления, 
# текстовые признаки, убрана колонка с текстом и нуикальные идентификаторы
import pandas as pd
# import numpy as np
from .utils import *

class Preprocessing:
	def __init__(self, data, cols_to_remove, text_col, label_col):
		self.data = data
		self.cols_to_remove = cols_to_remove
		self.text_col = text_col
		self.label_col = label_col
	def get_data(self):
		return self.data
	def remove_cols_drop_duplicated(self):
		self.data.drop(self.cols_to_remove, axis=1, inplace=True)
		self.data = self.data[self.data.duplicated() == False].reset_index(drop=True)
		return self
	def add_division(self):
		#cols - список числовых параметров
		cols = list(self.data.columns)
		cols.remove(self.text_col)
		if len(self.label_col) > 0:
			cols.remove(self.label_col)
		for i in range(len(cols)):
		    for j in range(i+1, len(cols)):
		        if len(self.data[cols[j]][self.data[cols[j]]==0].values) == 0:
		            name = 'div_' + cols[i] + '_' + cols[j]
		            self.data[name] = self.data[cols[i]] / self.data[cols[j]]
		return self
	def add_text_params(self):
		self.data['num_points'] = self.data[self.text_col].apply(lambda x: x.count(',') + x.count(';') \
                                   + x.count(':')+ x.count('.')+ x.count('!')+ x.count('?')\
                                   + x.count('"')+ x.count("'")+ x.count('/'))
		self.data[self.text_col] = self.data[self.text_col].apply(lambda x: tokenizer(x))
		self.data['num_big_letters'] = self.data[self.text_col].apply(lambda x: big_letter(x))
		self.data[self.text_col]= self.data[self.text_col].apply(lambda x: x.lower())
		self.data['avg_word_len'] = self.data[self.text_col].apply(avg_word_len)
		self.data['num_words'] = self.data[self.text_col].apply(lambda x: len(x.split()))
		self.data['avg_big_letters'] = self.data.num_big_letters / self.data.num_words
		self.data['avg_points'] = self.data.num_points / self.data.num_words
		self.data = self.data.fillna(0)
		return self
	def drop_text(self):
		self.data.drop(self.text_col, axis=1, inplace=True)
		return self
	def full_preprocessing(self):
		return self.remove_cols_drop_duplicated().add_division().add_text_params().drop_text().get_data()
