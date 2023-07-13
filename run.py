import src

GlobalConfig = src.config.GlobalConfig

src.make_processed_data(
	GlobalConfig.DATA_PATH, 
	GlobalConfig.RAW_DATA_PATH, 
	GlobalConfig.PROCESSED_DATA_PATH, 
	GlobalConfig.TRAIN_DATA_PATH, 
	GlobalConfig.INFERENCE_DATA_PATH,
	GlobalConfig.DATA_NAME, 
	GlobalConfig.MODE, 
	GlobalConfig.LABEL_COLUMN, 
	GlobalConfig.TEXT_COLUMN, 
	GlobalConfig.IDS
	)
if GlobalConfig.MODE == 0:
	src.train_model(
		GlobalConfig.DATA_PATH, 
		GlobalConfig.PROCESSED_DATA_PATH, 
		GlobalConfig.TRAIN_DATA_PATH,
		GlobalConfig.DATA_NAME, 
		GlobalConfig.LABEL_COLUMN,
		GlobalConfig.MODEL_PATH,
		GlobalConfig.MODEL_NAME
		)

if GlobalConfig.MODE == 1:
	src.predict_model(
		GlobalConfig.DATA_PATH, 
		GlobalConfig.PROCESSED_DATA_PATH, 
		GlobalConfig.INFERENCE_DATA_PATH,
		GlobalConfig.DATA_NAME, 
		GlobalConfig.MODEL_PATH,
		GlobalConfig.MODEL_NAME,
		GlobalConfig.PREDICTIONS_NAME
		)

if GlobalConfig.MODE == 2:
	src.train_model(
		GlobalConfig.DATA_PATH, 
		GlobalConfig.PROCESSED_DATA_PATH, 
		GlobalConfig.TRAIN_DATA_PATH,
		GlobalConfig.DATA_NAME, 
		GlobalConfig.LABEL_COLUMN,
		GlobalConfig.MODEL_PATH,
		GlobalConfig.MODEL_NAME
		)
	src.predict_model(
		GlobalConfig.DATA_PATH, 
		GlobalConfig.PROCESSED_DATA_PATH, 
		GlobalConfig.INFERENCE_DATA_PATH,
		GlobalConfig.DATA_NAME, 
		GlobalConfig.MODEL_PATH,
		GlobalConfig.MODEL_NAME,
		GlobalConfig.PREDICTIONS_NAME
		)
