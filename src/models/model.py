# Модель с подобранными параметрами
from sklearn.linear_model import LogisticRegression
from .. import config
class Model:
	def initialize_model(self):
		return LogisticRegression(
			C = config.Parameters.C, 
			class_weight = config.Parameters.class_weight,
			dual = config.Parameters.dual,
			fit_intercept = config.Parameters.fit_intercept,
			intercept_scaling = config.Parameters.intercept_scaling,
			max_iter = config.Parameters.max_iter, 
			penalty = config.Parameters.penalty,
			solver = config.Parameters.solver, 
			tol = config.Parameters.tol,
			warm_start = config.Parameters.warm_start
			)