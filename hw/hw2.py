import tensorflow as tf
import numpy as np
from tensorflow import keras

def func(x):
    shape = x.shape[0]
    f = 0
    for i in range(shape-1):
        f += 100*((x[i+1]-x[i]**2)**2+(1-x[i])**2)
    return f
    
def Laplacian(func, x):
    with tf.GradientTape() as g:
        with tf.GradientTape() as gg:
            y = func(x)
        dy_dx = gg.gradient(y, x) 
    d2y_dx2 = g.gradient(dy_dx, x) 
    return np.sum(d2f_d2x)

if __name__ == "__main__":
    x = tf.Variable(np.random.randn(100,1))
    print(Laplacian(func, x))