import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses

# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"



class My_SGD():

    def __init__(self,
                loss_func = None,
                nn = None,
                learning_rate=0.01,
                momentum=0.99,
                name="SGD",
                epoch = 0,
                batch_size = 64,
                Nesterov=False,
                Adam=False,
                beta1=0.9,
                beta2=0.999,
                eps=1e-6,
                t=1,
                **kwargs):

        # super(SGD, self).__init__(name, **kwargs)
        self.learning_rate = learning_rate
        self.name = name
        self.momentum = momentum
        self.epoch = epoch
        self.net = nn
        self.loss = loss_func
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = t

        
    def update_para(self, para, grads, state_v = None):
        for i,p in enumerate(para):
            p.assign_sub(self.learning_rate * grads[i])
        

    def update_para_momentum(self, para, grads, state_v = None):
        i = 0

        for p,v in zip(para, state_v):
            v = tf.cast(v, dtype = tf.float64)
            v = self.learning_rate * grads[i] + self.momentum * tf.squeeze(v)
            p.assign_sub(v)
            i+=1

    def init_momentum_states(self, features):
        v_w = tf.zeros((features.shape[-1], 1))
        v_b = tf.zeros(1)
        return (v_w, v_b)

    def init_adam_states(self, features):
        v_w, v_b = np.zeros((features.shape[-1]), dtype=float), np.zeros(1, dtype=float)
        s_w, s_b = np.zeros((features.shape[-1]), dtype=float), np.zeros(1, dtype=float)
        return ((v_w, s_w), (v_b, s_b))

    def adam(self, para, grads, states = None):
        # beta1, beta2, eps, i = 0.9, 0.999, 1e-6, 0
        i = 0
        for p, (v, s) in zip(para, states):
            v[:] = self.beta1 * v + (1 - self.beta1) * grads[i]
            s[:] = self.beta2 * s + (1 - self.beta2) * grads[i]**2
            v_bias_corr = v / (1 - self.beta1 ** self.t)
            s_bias_corr = s / (1 - self.beta2 ** self.t)
            p.assign_sub(self.learning_rate * v_bias_corr / (np.sqrt(s_bias_corr) + self.eps))
            i+=1
        self.t += 1



def train(x = [tf.Variable(np.random.rand(10))], epochs=20000, lr = 0.001, momentum = 0, Adam = False, Nesterov = False):

    optimizer = My_SGD(learning_rate=lr)



    if Adam == False:

        if momentum == 0:
            update_func = optimizer.update_para
            state = None

        else:
            update_func = optimizer.update_para_momentum
            state = optimizer.init_momentum_states(x[0])

    else:
        state = optimizer.init_adam_states(x[0])
        update_func = optimizer.adam

    history = {}

    history['f_list'] = []
    history['epochs'] = epochs
    for epoch in range(epochs):
        if Nesterov == False:
            with tf.GradientTape() as tape:
                f = func(x[0])

            grads = tape.gradient(f, x)
        else:
            state_N = tf.cast(state[0], dtype = tf.float64)
            y = x[0] + momentum * tf.squeeze(state_N)
            with tf.GradientTape() as tape:   
                tape.watch(y)             
                f = func(y)

            grads = tape.gradient(f, y)
            # print(grads)

        update_func(x, grads, state)

        history['f_list'].append(f)

        template = 'Epoch {}, Loss: {}'
        print (template.format(epoch+1, f))

    return history

    
def save_history(history, dirs = './result', name = 'sgd'):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    np.savez(dirs+'/' + name, epochs =history["epochs"], f_list = history["f_list"])

def plot_history(history, name = 'sgd', dirs = './figure'):

    plt.plot(range(history["epochs"]),  history["f_list"], color='blue', label=name)

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')

    plt.legend()

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    plt.savefig(dirs+'/' + name+'.png')

    plt.close()


    




def func(x=None):
    shape = x.shape[0]
    l = 0
    for i in range(shape-1):
        l += (100*(x[i+1]-x[i]**2)**2+(1-x[i])**2)


    return l





if __name__ == "__main__":

    # history_sgd = train()
    # plot_history(history_sgd)

    # history_momentum = train(momentum=0.99)
    # plot_history(history_momentum, name = 'momentum')

    history_adam = train(Adam = True)
    plot_history(history_adam, name = 'adam')

    # history_momentum = train(lr=0.0001,momentum=0.99, Nesterov=True)


    
