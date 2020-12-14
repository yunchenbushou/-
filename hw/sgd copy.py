import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses

# from tensorflow.python.keras.optimizer_v2 import optimizer_v2

os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"



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
        self.Nesterov = Nesterov
        self.Adam = Adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = t
        self.history = {}

        
    def update_para(self, para, grads, state_v = None):
        for i,p in enumerate(para):
            p.assign_sub(self.learning_rate * grads[i])
        

    def update_para_momentum(self, para, grads, state_v = None):
        # for i,p in enumerate(para):
        i = 0
        # print(len(para))

        for p,v in zip(para, state_v):
            v = tf.cast(v, dtype = tf.float64)
            v = self.learning_rate * grads[i] + self.momentum * tf.squeeze(v)
            # print(v)
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

    # def train_loop(self, features, labels, para=None, state=None):

    #     loss_obj = tf.keras.losses.mean_squared_error

    #     train_loss = tf.keras.metrics.Mean(name='train_loss')

    #     data_iter = tf.data.Dataset.from_tensor_slices((features,labels)).batch(self.batch_size)
    #     data_iter = data_iter.shuffle(100)
    #     # if para == None:
    #     #     para = self.net.trainable_weights
    #     # para = tf.compat.v1.trainable_variables()
    #     history = {}
    #     history["train_loss"] = []
    #     # history["test_loss"] = []
    #     history["epoch"] = []

    #     if self.Adam == False:

    #         if self.momentum == 0:
    #             update_func = self.update_para

    #         else:
    #             update_func = self.update_para_momentum
    #             state = self.init_momentum_states(features)

    #     else:
    #         state = self.init_adam_states(features)
    #         update_func = self.adam


        
    #     for epoch in range(self.epoch):

    #         # loss_obj.reset_states()
    #         train_loss.reset_states()

    #         for batch_i, (X, y) in enumerate(data_iter):
    #             with tf.GradientTape() as tape:
    #                 l = loss_obj(self.net(X), y)
                    

    #             if para == None:
    #                 para = self.net.trainable_weights

    #             if self.Nesterov == False:
    #                 grads = tape.gradient(l, para)
    #             else:
    #                 grads = tape.gradient(l, para + self.momentum * state)
    #             print(grads)
    #             update_func(para, grads, state)
    #             train_loss(l)

    #         template = 'Epoch {}, Loss: {}'
    #         print (template.format(epoch+1, train_loss.result()))
    #         history["train_loss"].append(train_loss.result().numpy())
    #         history["epoch"].append(epoch+1)

    #     self.history = history
    #     # self.net.save_weights("./model/saved_model.h5")

def train(epochs=1000, lr = 0.0001, momentum = 0, Adam = False, Nesterov = False):
    x = [tf.Variable(np.random.rand(100))]

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

    # state = optimizer.init_adam_states(x[0])

    history = {}

    history['f_list'] = []
    history['epochs'] = epochs
    for epoch in range(epochs):

        with tf.GradientTape() as tape:
            f = func(x[0])

        if Nesterov == False:
            grads = tape.gradient(f, x)
        else:
            # print(state)
            tmp[0] = 
            grads = tape.gradient(f, x[0] + momentum * state)

        # grads = tape.gradient(f, x)
        update_func(x, grads, state)

        history['f_list'].append(f)

        template = 'Epoch {}, Loss: {}'
        print (template.format(epoch+1, f))

    return history

    
def save_history(history, dirs = './result', name = 'sgd'):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    np.savez(dirs+'/' + name, epochs =history["epoch"], f_list = history["f_list"])

def plot()



    




def func(x=None):
    shape = x.shape[0]
    l = 0
    for i in range(shape-1):
        l += 100*((x[i+1]-x[i]**2)**2+(1-x[i])**2)


    return l





if __name__ == "__main__":

    train()



    
