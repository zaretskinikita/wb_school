# Задача разделения фейковых и реальных комментариев Wildberries
В репозитории представлен анализ данных и процесс разработки решения, а также готовый проект.

<!-- данные -->
## 1. Данные
Для разделения используются данные в формате: 3 уникальных идентификатора, 8 числовых признаков и текст комментария (в случае обучения - еще колонка с метками).

<!-- модель -->
## 2. Модель
Используется модель логистической регрессии. 

<!-- файлы -->
## 3. Файлы
* **requirements.txt**: Список необходимых библиотек для запуска проекта.

* **src**: Содержит программы для запуска проекта. 
    * **src/config**: Содержит все настраиваемые параметры (class Parameters для гиперпараметров модели, класс GlobalConfig для проекта (пути к скриптам, данным, модели и предсказаниям модели с учетом структуры проекта, название столбца с метками, названия столбца с текстом, названия столбцов с уникальными id, режим работы проекта (Train(0)/Inference(1)/Both(2)))).
     **src/logger**: Содержит настройки для логгирования.
    * **src/features**: Содержит скрипты для предобработки данных
        * **features/utils**: Вспомогательные функции для предобработки текста.
        * **features/preprocessing**: Содержит класс Preprocessing, осуществляющий предобработку данных для обучения модели. Обогащение исходных данных попарными делениями числовых признаков (кроме деления на признаки, в которых есть 0) и текстовыми признаками: количество знаков препинания, количество слов, начинающихся с большой буквы, средняя длина слова в комментарии, количество слов, среднее количество слов, начинающихся с большой буквы и среднее количество знаков препинания в пересчете на количество слов. 
        * **features/data_features**: Содержит функцию make_processed_data(), осуществляющую предобработку данных и их сохранение в ./data/ в зависимости от режима (Train/Inference).
    * **src/models**: Содержит скрипты для работы модели.
        * **features/params**: Содержит класс Parameters, содержащий параметры модели и порог.
        * **features/model**: Содержит класс Model, осуществляющий инициализацию модели.
        * **features/train_model**: Содержит функцию train_model(), осуществляющую обучение модели на размеченных данных и ее сохранение в папку ./models.
        * **features/data_features**: Содержит функцию predict_model(), осуществляющую применение модели к неразмеченных данных и сохранение результатов в ./models.

* **data**: Содержит сырые и предобработанные данные 
    * **raw/train/**: Содержит сырые данные для обучения (размеченные). Перед обучением они должны быть загружены. 
    * **raw/inference/**: Содержит сырые данные для применения (неразмеченные). Перед применением они должны быть загружены. 
    * **processed/train/**: Содержит данные для обучения после предобработки.
    * **processed/inference/**: Содержит данные для применения после предобработки.
    * **data_analysis.csv**: Данные, используемые для анализа и разработки (см. ./notebooks/)
    
* **models**: Содержит результат обучения (модель) и результат применения (предсказания модели).

* **reports**: Содержит данные для отчетности.

* **notebooks**: Содержит Jupyter Notebooks, показывающие развитие проекта. 
    * **1-zaretskii-EDA_L0-L3.ipynb**: Содержит EDA, формализацию задачи, анализ исходных данных.
    * **2.1-zaretskii-baseline-and-models-L4.ipynb**: Содержит формирование baseline, подбор и проверку возможных решений на простых моделях.
    * **2.2-zaretskii-Text_CNN.ipynb**: Содержит формирование ансамбля CNN, обученной на предобработанном с помощью BERT тексте, и модели, обученной на числовых признаках.
    * **3-zaretskii-evaluation-tuning-L5.ipynb**: Содержит подбор гиперпараметров для выбранных решений и выбор конечной модели. 
    * **4-zaretskii-chosen_model_NoOutliers.ipynb**: Содержит обучение выбранной модели на данных без выбросов.
    
* **run.py**: Файл для запуска модели с параметрами из ./src/config.  

<!-- how-to-use -->
## 3. Использование
Установка нужных расширений:

```SH
pip install -r requirements.txt
```

Conda:
```SH
conda install pip
pip install -r requirements.txt
```



Параметры можно вручную изменить в файле ./src/config

```PY
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
```

Пример запуска модели: 

```SH
python3 run.py 
```
