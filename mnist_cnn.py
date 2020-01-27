# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Класс классифицирующий рукописные числа на картинке размеров 28 х 28
# Для обучения используется датасет MNIST (http://yann.lecun.com/exdb/mnist/)
class TextRecognizer:

    def __init__(self, load=False):
        #
        # Загрузка датасета
        #
        # train_images - Данные для обучения. 3-ех мерная стркутура, содержащая
        #                данные о пикселях 60000 картинок размером 28х28
        #
        # train_labels - Данные о содержании картинок для обучения (метки, лейблы).
        #
        # test_images - Данные для тестирования. 3-ех мерная стркутура, содержащая
        #               данные о пикселях 10000 картинок размером 28х28
        #
        # test_labels - Данные о содержании картинок для тестирования (метки, лейблы).
        #
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.load_dataset()

        if load:
            # Загрузка готовой, обученной модели
            self.model = load_model('mnist_model.h5')
        else:
            # Создание пустой модели
            self.model = self.create_empty_model()
            # Обучение модели на тренировочных данных
            self.model.fit(self.train_images, self.train_labels, epochs=10, verbose=1)
            # Сохранение данных модели
            self.model.save('mnist_model.h5')

        # Расчет точности на тестовых данных
        loss, acc = self.model.evaluate(self.test_images, self.test_labels, verbose=3)
        print("Точность модели: {:5.2f}%".format(100 * acc))

        weights_, biases_ = self.model.get_layer(index=0).get_weights()
        weights, biases = self.model.get_layer(index=2).get_weights()

        w = weights_.dot(weights).reshape(28, 28, 10)
        print(w.shape)
        w = np.moveaxis(w, -1, 0)

        plt.figure(figsize=(5, 5))

        for i in range(10):
            print(w[i].shape)
            plt.subplot(5, 2, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(w[i])
            plt.xlabel(str(i))
        plt.show()

    # Метод, загружающий датасет
    def load_dataset(self):
        # Загружаем датасет через готовую функцию из библиотеки Keras
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        # Для ускорения обучения используем только 1000 картинок
        train_labels = train_labels[:1000]
        test_labels = test_labels[:1000]

        # Изменяем размерности 3-х мерных структур (1000 х 28 х 28) в двухмерные (1000 х 784) для упрощения
        train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
        test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

        return (train_images, train_labels), (test_images, test_labels)

    # Метод, создающий необученную модель нейросети с помощью библиотеки Keras
    def create_empty_model(self):
        # Sequential - линейный набор слоев
        relu_layer = Dense(512, activation='relu', input_shape=(28 * 28,))
        softmax_layer = Dense(10, activation='softmax', name='last')

        model = Sequential([
            # Слой типа Dense (все нейроны слоя связаны со всеми нейронами следуюзего слоя),
            # в качестве активационной функции используется ReLU (Rectified Linear Unit)
            relu_layer,
            # Dropout удаляет случайные нейронов с заданной вероятностью (метод регуляризации)
            Dropout(0.2),
            # Слой типа Dense, в качестве активационной функции используется Softmax
            softmax_layer
        ])
        # Собираем модель, в качестве оптимизатора используем стохастический градиентный спуск
        # (Stochastic Gradient Descent, SGD) и в качестве функции потерь перекресную энтропию (Cross Entropy)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Функция прогнозирования класса картинки
    def predict(self, file):
        model = self.model

        # Загружаем изображение в монохромном формате
        test_img = image.load_img(file, target_size=(28, 28), color_mode='grayscale')
        # Преобразуем его в числа
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)

        # Изменяем размерность структуры в подходящую нам
        test_img = test_img.reshape(-1, 28 * 28) / 255.0

        # Классифицируем изображение на основе полученных данных при обучении
        return model.predict_classes(test_img, batch_size=1)

