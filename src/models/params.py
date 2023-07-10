#Подобранные параметры модели и порог отбора
class Parameters:
   threshold : float = 0.7
   C: int = 1
   class_weight: str = 'balanced' 
   dual: bool = False
   fit_intercept: bool = False 
   intercept_scaling: int =  1
   max_iter: int = 1000 
   penalty: str = 'l1'
   solver: str = 'liblinear' 
   tol: float = 0.001
   warm_start: bool = False
