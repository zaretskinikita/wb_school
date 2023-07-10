# Вспомогательные функции для формирования текстовых признаков в файле Preprocessing

from string import digits, whitespace
import re
import numpy as np
def tokenizer(text):
	'''
	Функция удаляет из текста все, кроме русских символов
	'''
	cyrillic_letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
	text = text.replace(',', ' ')
	text = text.replace(':', ' ')
	text = text.replace(';', ' ')
	text = text.replace('.', ' ')
	text = text.replace('!', ' ')
	text = text.replace('?', ' ')
	text = text.replace('"', ' ')
	text = text.replace("'", ' ')
	text = text.replace('/', ' ')
	allowed_chars = cyrillic_letters + digits + whitespace
	return re.sub("\s\s+", " ", "".join([c for c in text if c in allowed_chars])).strip()

def big_letter(x):
	'''
	Функция считает количество слов, начинающихся с большой буквы
	'''
	counter = 0
	for word in x.split():
	    if word[0].upper() == word[0]:
	        counter+=1
	return counter

def avg_word_len(x): 
	'''
	Функция считает среднюю длину слова в предложении
	'''
	x = x.split()
	lens = []
	if len(x) == 0:
		return 0
	for i in x:
		lens.append(len(i))
	return np.mean(lens)