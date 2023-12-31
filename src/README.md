# Задача разделения фейковых и реальных комментариев Wildberries

* **src**: Содержит программы для запуска проекта. 
    * **src/config**: Содержит все настраиваемые параметры (class Parameters для гиперпараметров модели, класс GlobalConfig для проекта (пути к скриптам, данным, модели и предсказаниям модели с учетом структуры проекта, название столбца с метками, названия столбца с текстом, названия столбцов с уникальными id, режим работы проекта (Train(0)/Inference(1)/Both(2)))).
    * **src/logger**: Содержит настройки для логгирования.
    * **src/features**: Содержит скрипты для предобработки данных
        * **features/utils**: Вспомогательные функции для предобработки текста.
        * **features/preprocessing**: Содержит класс Preprocessing, осуществляющий предобработку данных для обучения модели. Обогащение исходных данных попарными делениями числовых признаков (кроме деления на признаки, в которых есть 0) и текстовыми признаками: количество знаков препинания, количество слов, начинающихся с большой буквы, средняя длина слова в комментарии, количество слов, среднее количество слов, начинающихся с большой буквы и среднее количество знаков препинания в пересчете на количество слов. 
        * **features/data_features**: Содержит функцию make_processed_data(), осуществляющую предобработку данных и их сохранение в ./data/ в зависимости от режима (Train/Inference).
    * **src/models**: Содержит скрипты для работы модели.
        * **features/params**: Содержит класс Parameters, содержащий параметры модели и порог.
        * **features/model**: Содержит класс Model, осуществляющий инициализацию модели.
        * **features/train_model**: Содержит функцию train_model(), осуществляющую обучение модели на размеченных данных и ее сохранение в папку ./models.
        * **features/data_features**: Содержит функцию predict_model(), осуществляющую применение модели к неразмеченных данных и сохранение результатов в ./models.