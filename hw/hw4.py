import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers, datasets, models, losses

# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


def plot_metrics(history):
    metrics =  ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric
        plt.subplot(1,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color='pink', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color='pink', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)


        plt.legend()

    plt.savefig('./figure/mnist.png')

def get_dataset():
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
    num_train = 1000
    num_test = 200
    train_x, train_y = slice_data(train_x, train_y, num_train)
    test_x, test_y = slice_data(test_x, test_y, num_test)

    train_x, test_x = train_x/ 255.0, test_x/ 255.0
    # train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    # train_ds = train_ds.shuffle(buffer_size=buffer_size).batch(batch_size)
    # test_ds = test_ds

    return (train_x, train_y), (test_x, test_y)




def slice_data(feature, labels, num):
    feature_list = []
    label_list =[]
    count_a, count_t = 0, 0

    for idx, label in enumerate(labels):

        if (label == 0) and (count_a < num):
            count_a += 1
            label_list.append(0)
            feature_list.append(feature[idx].astype(np.float32))
        elif (label == 9) and (count_t < num):
            count_t += 1
            label_list.append(1)
            feature_list.append(feature[idx].astype(np.float32))

        if (count_a == num) and (count_t == num):
            break
    # print(feature_list.shape)
    return np.array(feature_list), np.array(label_list)


def train():

    (train_x, train_y), (test_x, test_y) = get_dataset()
    # print(train_y == 0)

    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3), padding = "SAME"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding = "SAME"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding = "SAME"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "SAME"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = "SAME"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='relu'))
    model.summary()

    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )


    history = model.fit(train_x, train_y, epochs=50, batch_size=32, validation_data = (test_x, test_y))

    # print(history.history)
    plot_metrics(history)

if __name__ == '__main__':
    train()
    # get_dataset()