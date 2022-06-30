import os
import numpy as np
from tensorflow import keras
from pyrsgis import raster
from pyrsgis.convert import changeDimension
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Используемые файлы
mxKrasnoyarsk = 'WV214JUN2014_raw.tif'
forestKrasnoyarsk = 'WV214JUN2014_1_2.tif'
mxTest = 'gmaps-lep.0.tif'

# Чтение растрового изображение в массив
ds1, featuresKrasnoyarsk = raster.read(mxKrasnoyarsk, bands='all')
ds2, labelKrasnoyarsk = raster.read(forestKrasnoyarsk, bands=1)
ds3, featuresTest = raster.read(mxTest, bands='all')

# Вывод на экран размера прочитанных данных
print("Bangalore Multispectral image shape: ", featuresKrasnoyarsk.shape)
print("Bangalore Binary built-up image shape: ", labelKrasnoyarsk.shape)
print("Hyderabad Multispectral image shape: ", featuresTest.shape)

# Изменяем значения NoData на ноль
labelKrasnoyarsk = (labelKrasnoyarsk == 1).astype(int)

'''
При помощи модуля convert из пакета pyrsgis меняем форму массивов на двумерный,
каждая строка которого будет представлять собой пиксель
'''
featuresKrasnoyarsk = changeDimension(featuresKrasnoyarsk)
labelKrasnoyarsk = changeDimension(labelKrasnoyarsk)
featuresTest = changeDimension(featuresTest)
nBands = featuresKrasnoyarsk.shape[1]

# Вывод на экран размера прочитанных данных
print("Красноярск мультиспектральная форма изображения: ", featuresKrasnoyarsk.shape)
print("Красноярск бинарная форма изображения: ", labelKrasnoyarsk.shape)
print("Тестовая мультиспектральная форма изображения: ", featuresTest.shape)

# Разделение на тестовый и обучающий датасеты в соотношении 60/40
xTrain, xTest, yTrain, yTest = train_test_split(featuresKrasnoyarsk, labelKrasnoyarsk, test_size=0.4, random_state=42)

print(xTrain.shape)
print(yTrain.shape)

print(xTest.shape)
print(yTest.shape)

'''
Нормализация данных путем вычитания минимального значения и деления на диапазон.
Поскольку данные Landsat представляют собой 8-битные данные, минимальное и максимальное значения равны 0 и 255
'''
xTrain = xTrain / 255.0
xTest = xTest / 255.0
featuresTest = featuresTest / 255.0

# Изменяем форму объектов с двухмерной на трехмерную
xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
featuresTest = featuresTest.reshape((featuresTest.shape[0], 1, featuresTest.shape[1]))

# Вывод формы измененных данных
print(xTrain.shape, xTest.shape, featuresTest.shape)

'''
Задаем параметры модели, используется последовательная модель Sequential.
Есть один входной слой с количеством узлов, равным nBands — в нашем случае их 4. 
Так же есть один скрытый слой с 14 узлами и `relu` в качестве функции активации.
Последний слой содержит два узла для бинарного построенного класса с функцией активации softmax, 
которая подходит для категориального вывода.
Элементы выходного вектора находятся в диапазоне (0, 1) и в сумме равны 1.
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(2, activation='softmax')])

# Определение метрик и параметров точности
'''
Компилируем модель с оптимизатором `adam`.
В качестве функции потерь перекрестную энтропию (`sparse_categorical_crossentropy`). 
Метрикой для оценки производительности модели является `accuracy`.
'''
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Запуск модели в течении двух эпох
model.fit(xTrain, yTrain, epochs=2)

'''
Функция softmax создает отдельные столбцы для каждого значения вероятности типа класса. 
Извлекаем только для класса один (лес).
'''
predicted = model.predict(featuresTest)
predicted = predicted[:, 1]

# Прогнозирование новых данных и экспорт растра вероятности
prediction = np.reshape(predicted, (ds3.RasterYSize, ds3.RasterXSize))
outFile = 'output_file.tif'
raster.export(prediction, ds3, filename=outFile, dtype='float')
