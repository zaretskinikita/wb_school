# Модель с подобранными параметрами
from sklearn.linear_model import LogisticRegression
from .params import Parameters
class Model:
	def initialize_model(self):
		return LogisticRegression(
			C = Parameters.C, 
			class_weight = Parameters.class_weight,
			dual = Parameters.dual,
			fit_intercept = Parameters.fit_intercept,
			intercept_scaling = Parameters.intercept_scaling,
			max_iter = Parameters.max_iter, 
			penalty = Parameters.penalty,
			solver = Parameters.solver, 
			tol = Parameters.tol,
			warm_start = Parameters.warm_start
			)