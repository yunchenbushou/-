import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve
from scipy.stats import norm
from tensorflow import keras
from tensorflow.keras import initializers

# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
class My_DNN(tf.Module):

    def __init__(self, input_size, depth, width, alpha, name=None, res_net=False):
        super(My_DNN, self).__init__(name=name)
        self.depth = depth
        self.input_size = input_size
        self.width = width
        self.alpha = alpha
        self.res = res_net

    def build_NN(self):
        if self.res:
            Inputs = keras.Input(shape=(self.input_size,))
            x = keras.layers.Dense(self.width, activation='relu', kernel_initializer=initializers.RandomNormal(
                stddev=tf.sqrt(self.alpha/self.width)))(Inputs) + Inputs
            for _ in range(self.depth-2):
                x = tf.keras.layers.Dense(self.width, activation='relu', kernel_initializer=initializers.RandomNormal(
                    stddev=tf.sqrt(self.alpha/self.width)))(x) + x
            Outputs = tf.keras.layers.Dense(1, activation='relu', kernel_initializer=initializers.RandomNormal(
                stddev=tf.sqrt(self.alpha/100)))(x)

            model = keras.Model(inputs=Inputs, outputs=Outputs)
            self.nn = model

        else:
            Inputs = keras.Input(shape=(self.input_size,))
            x = keras.layers.Dense(self.width, activation='relu', kernel_initializer=initializers.RandomNormal(
                stddev=tf.sqrt(self.alpha/self.width)))(Inputs)
            for _ in range(self.depth-2):
                x = tf.keras.layers.Dense(self.width, activation='relu', kernel_initializer=initializers.RandomNormal(
                    stddev=tf.sqrt(self.alpha/self.width)))(x)
            Outputs = tf.keras.layers.Dense(
                1, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=tf.sqrt(self.alpha/100)))(x)

            model = keras.Model(inputs=Inputs, outputs=Outputs)
            self.nn = model

    def __call__(self, x):
        x = self.nn(x)
        return x

def delta(input_size, alpha, res_net=False, loss=tf.keras.losses.mean_squared_error):
    train_x = tf.random.normal([10000, input_size], mean=0, stddev=1)
    train_y = 1.0

    model = My_DNN(input_size, 100, 100, alpha, res_net=res_net)
    model.build_NN()

    with tf.GradientTape() as tape:
        l = tf.reduce_mean(loss(model(train_x), train_y))

    grads = tape.gradient(l, model.nn.trainable_weights)

    # print(grads)
    x = range(1, 101)
    y = [np.linalg.norm(np.array(x), ord='fro') for x in grads[::2]]
    plt.plot(x, y)

    plt.xlabel('depths')
    plt.ylabel('fro norm of delta')
    plt.title('alpha=%s'% alpha)
    if res_net:
        plt.savefig('./figure/resnet_delta_alpha=%s.png' % alpha)
    else:
        plt.savefig('./figure/delta_alpha=%s.png' % alpha)

    plt.close()




def Numerical_sol(activations = 'relu'):
    x=np.random.normal(size=100000)
    def func(alpha):
        if activations == 'relu':
            sol = np.mean(keras.activations.relu(alpha*x)**2) - 1
        if activations == 'swish':
            sol = np.mean(keras.activations.swish(alpha*x)**2) - 1
        return sol

    root = fsolve(func, 1)

    return root


if __name__ == "__main__":
    alpha = [1, 1.5, 2, 2.5]
    for i in alpha:
        delta(100, i)
    alpha = [0.5, 1, 1.5, 2]
    for i in alpha:
        delta(100, i, res_net=True)

    print(Numerical_sol(activations='swish'))
    print(Numerical_sol(activations='relu'))
