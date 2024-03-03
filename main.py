import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_data, y_data), (x_test, y_test) = mnist.load_data()
x_data = x_data / 255.0
x_test = x_test / 255.0
x_data_list = []
y_data_list = []
optimizer = keras.optimizers.Adam(learning_rate=0.001)
k = 5
epoch = 1
batch_size=32

model = keras.Sequential(layers=[
    # 첫 Filter 개수가 32개인 이유는 처음부터 64개로 하면 이미지가 너무 커서 연산량이 너무 많기 때문
    # Padding으로 인해 28*28 -> 28*28
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu', padding="same"),
    # Pooling 연산으로 인해 28*28 -> 14*14
    keras.layers.MaxPooling2D((2,2)),
    # Conv 연산으로 인해 14*14 -> 12*12
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    # Pooling 연산으로 인해 12*12 -> 6*6
    keras.layers.MaxPooling2D((2,2)),
    # Conv 연산으로 인해 6*6 -> 4*4
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()
# Training, Validation 분할
for i in range(k):
    interval = int(len(x_data) / k)
    start_idx = i * interval
    end_idx = (i+1) * interval
    x_data_list.append(x_data[start_idx : end_idx])
    y_data_list.append(y_data[start_idx : end_idx])

# K-Fold Validation
for i in range(k):
    # validation data
    x_val = np.array(x_data_list.pop(0))
    y_val = np.array(y_data_list.pop(0))

    x_train = []
    y_train = []

    # use the rest of data to train
    for j in range(k-1):
        x_train.extend(x_data_list[j])
        y_train.extend(y_data_list[j])
    
    x_train = np.array(x_train)
    x_train = x_train.reshape(int(60000/k * (k-1)), 28, 28, 1)
    print(x_train.shape)
    y_train = np.array(y_train)
    #print(x_train[0])
    history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_val, y_val))
    #print(history)
    #for j in range(1):
        #reshape 중요
        #x_t = x_train[0].reshape(1, 28, 28, 1)
        
        
     #   print(result.shape)

    # reuse as train data
    x_data_list.append(x_val)
    y_data_list.append(y_val)

    if i == k-1:
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
        print(test_loss)
        print(test_acc)
