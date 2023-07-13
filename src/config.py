from pydantic import BaseSettings
from pathlib import Path
import os


class GlobalConfig(BaseSettings):
    #PROJECT_PATH: Path to the project (automatically defined)
    PROJECT_PATH: Path = Path(
      os.path.abspath(
         os.path.join(
            os.path.dirname(
               os.path.realpath(__file__)), os.pardir
            )
         )
      )

    # DATA_PATH: Path to the data folder
    DATA_PATH: Path = PROJECT_PATH / 'data'
    # RAW_DATA_PATH: Path to the raw data folder
    RAW_DATA_PATH: Path = 'raw'
    # PROCESSED_DATA_PATH: Path to the processed data folder
    PROCESSED_DATA_PATH: Path = 'processed'
    # TRAIN_DATA_PATH: Train data folder
    TRAIN_DATA_PATH: Path = Path('train')
    # INFERENCE_DATA_PATH: Inference data folder
    INFERENCE_DATA_PATH: Path = Path('inference')
    # MODEL_PATH: Path to the saved model (after training)
    MODEL_PATH: Path = PROJECT_PATH / 'models'

    # MODE: Train (0) / Inference (1) mode / Default (2) (both train and inference)
    MODE: int = 2
    # DATA_NAME: Name of the raw data file
    DATA_NAME: str = 'data.csv'
    # LABEL_COLUMN: Name of the column with labels (when training)
    LABEL_COLUMN: str = 'label'
    # TEXT_COLUMN: Name of the text column
    TEXT_COLUMN: str = 'text'
    # MODEL_NAME: Name of the trained model
    MODEL_NAME: str = 'LogRegr.pkl'
    # PREDICTIONS_NAME: Name of file with predictions (after inference)
    PREDICTIONS_NAME: str = "LogRegrPredictions.pkl"
    # IDS: List of Ids to remove
    IDS: list = ['id1', 'id2', 'id3'] 
    

#Подобранные параметры модели и порог отбора
class Parameters(BaseSettings):
   threshold: float = 0.7
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


GlobalConfig = GlobalConfig()
Parameters = Parameters()








