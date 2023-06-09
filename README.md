# Проект "Прогнозирование стоимости автомобиля по характеристикам, описанию и фотографии" (car_price). 
## Project 8 of the course "Specialization Data Science" that updated to the app 

# Структура работы над проектом
I) EDA (в отдельном Fork'e)

II) Построение "наивной"/baseline модели, предсказывающую цену по модели и году выпуска (для сравнения с другими моделями)  

III) Обработка и нормировка признаков -> получение таблицы данных для подачи в ML и DL - модели  

IV) Создание модели ML с помощью CatBoost (градиентный бустинг), тренировка, оценка результата    

V) Создание моделей DL:

Va) SimpleNN - Создание простой модели DL (на основе полносвязной нейронной сети), тренировка, оценка результата  

Vb) NLP:
- Работа с текстом, обработка приведение в векторный вид  
- NLP - Cоздание модели DL для работы с текстом (блоки LSTM, GRU, Transformer)  

Vc) Создание multi-input нейронной сети (SimpleNN + NLP) для анализа табличных данных и текста одновременно, тренировка, оценка результата  

Vd) EFN:
- обработка изображений с помощью библиотеки albumentations
- EFN - Cоздание модели DL для работы с изображениями на основе TransferLearning + использование техники FineTunning

Ve) Создание multi-input нейронной сети (SimpleNN + NLP + EFN) для анализа табличных данных, текста и картинок одновременно, тренировка, оценка результата 

VI) Ансамблирование градиентного бустинга и нейронной сети (усреднение их предсказаний)


**Задача** создать модель, которая будет предсказывать стоимость автомобиля. В качестве исходных данных были табличные данные с характеристиками автомобилей, описанием продавцов авто и фотографии.  
Целью данной работы было познакомиться с возможностью построения модели с использованием разного типа данных (Multi-Input сеть). 

Набор данных можно скачать: kaggle competitions download -c sf-dst-car-price-prediction-part2 либо по ссылке https://www.kaggle.com/c/sf-dst-car-price-prediction-part2/data 

Для работы с моделью в первую очередь была оценена "наивная" (Модель 1) и baseline модели. Наивная модель предсказывает среднюю цену по модели автомобиля и году выпуска. Данные модели необходимы для оценки качества более сложных моделей.  

Далее был проведен анализ табличных категориальных и числовых признаков (EDA). Анализ проведен в отдельном расшаренном файле https://www.kaggle.com/ostrebko/prolect-8-batmobil-eda. Также ноутбук можно скачать в данном проекте (prolect-8-batmobil-eda.ipynb).  

Следующим шагом была проведена обработка и нормировка признаков, в чтом числе на основе EDA (см. выше). Результатом работы стало получение таблицы данных для подачи в ML и DL - модели. В приведенном ноутбуке имеется строка с неактивным кодом, чтобы показать, какие новые признаки были созданы, однако их добавление не улучшило качество модели catboost (см. далее по тексту). При работе также были написаны отдельные функции для преобразования данных, чтобы можно было в последствии заново сгенерировать признаки, если потребуется возвращение к версии без изменений.

Для оценки качества моделей использовала метрика MAPE (Mean absolute percentage error) или Средняя абсолютная ошибка в процентах (см. https://en.wikipedia.org/wiki/Mean_absolute_percentage_error). 

Модель 2: Следующим шагом было создание модели "классического"  ML с помощью CatBoost (на основе градиентного бустинга), ее тренировка, оценка результата. Лучший полученный результат на тренировочной выборке и тестовой выборке составил 10,99%. Результат на валидационной выборке (Public Score), по которой оценивался submit был несколько хуже и составил 11.96%. 

Модель 3 представляет собой простую нейронная сеть с Dense слоями. В отличие от Baseline был добавлен слой BatchNormalization, изменен параметр units выходов в слоях Dense и слоев Dropout. Добавление большего числа скрытых слоев качество модели не улучшило. Лучший полученный результат на тренировочной выборке составил 10,29%, что гораздо лучше, чем результаты наивной модели, Baseline и Catboost. Результат на валидационной выборке (Public Score), по которой оценивался submit был несколько хуже, но все же был достаточно высоким и составил 10.93%.

Модель 4 представляет собой с две нейронные сети, выхода которых объединяются в единую нейронную сеть (добавляется одна "голова"). Каждая нейронная сеть (без "головы") предназначена обработки разных типов данных: табличные данные и текст. Для работы с табличными данными использована Model 3 (см. выше). Для работы с текстовыми данными была использована нейронная сеть с ячейками LSTM. Текст перед подачей в нейронную сеть "очищался" от знаков препинания, чисел, пробелов и повторяющихся больше 2-х раз букв, а затем каждое слово приводилось в нормальную форму с помощью библиотеки pymorphy2 (MorphAnalyzer), исключались "стоп-слова" (библиотека ntlk, stopwords). Дополнительно был добавлен фильтр с исключением часто и редко встречающихся слов. Далее обработанный текст токенизировался и векторизовался для последующей подачи в нейронную сеть. При работе нейронной сетью для обработки текста также в отличие от Baseline был добавлен слой BatchNormalization, изменен параметр units выходов в слоях Dense и слоев Dropout. Добавление большего числа скрытых слоев качество модели не улучшило. Лучший полученный результат на тренировочной выборке составил 10.19% (в отдельном случае был достигнут результат 10.12%). Качество модели оказалось лучше, чем результаты наивной модели, Baseline и Модели 3, при этом лучший Результат на валидационной выборке, по которой оценивался submit, составил 10.85% (10,80%), а добавление округления предсказания модели до 4 знака ДО запятой улучшило результаты на 0,1 процент.  

Проанализировано качество для Модели 5, которая строилась аналогично Модели 4 с добавлением нейронной сети EfficientNetB3 (Tranfer Learning) для анализа изображений (фото объявлений) с единой "головой" для трех подсетей. Для картинок в качестве предварительной обработки проводилась аугментация и были опробованы разные преобразования. Качества модели на текстовой выборке  достигалось качество до 10,44-10,49% (при округлении предсказания модели до 4 знака ДО запятой был примерно на 0,1 процент лучше, чем использование предсказаний модели без дополнительных преобразований), при этом результат на валидационной выборке (Public Score), по которой оценивался submit составил 11,23-11,26%. Были опробованы разные аугментации и разные скорости уменьшения learning rate с помощью callback (ReduceLROnPlateau) при обучении модели. Использование моделей, дающих лучшее качество на задаче ImageNet улучшения результата для используемой модели не дали (возможно в данном случае нужно было попробовать еще какие-нибудь различные конфигурации моделей, но на это не хватило времени).

Также было попробовано ансамблирование (простое голосование) градиентного бустинга и нейронных сетей. При ансаблировании лучших результатов всех 4-х моделей (Модель 2, Модель 3, Модель 4, Модель 5) был достигнут Public Score = 11,64. При ансаблировании лучших результатов 3-х моделей нейронных сетей (Модель 3, Модель 4, Модель 5) был достигнут Public Score = 11,626, что и является лучшим результатом, полученным в результате работы.

### Выводы по работе:
1. Была проведена работа по обработке данных, в т.ч. преобразовании текстовых для анализа модели. 
2. В процессе работы над моделью познакомился с созданием сети, состоящей из нейронных подсетей, которые на вход принимают разного рода данные (табличные данные, текст и изображения), обрабатывают и передают данные в слой, который объединяет данные и делает предсказание. Объединение нейронных сетей с различными типами данных может улучшить предсказание модели, однако это также сильно увеличивает возможное количество изменяемых параметров. Для работы с большим проектом уже нужна команда, которая будет работать над улучшением отдельного блока/части модели.
3. Большая часть преобразований по улучшению модели были опробованы, за исключением проброса признаков и проведения анализа результатов, для понимания, где модель ошибается. Также можно также попробовать более сложные модели обработки текста, например, transfer learning Bert'a.


