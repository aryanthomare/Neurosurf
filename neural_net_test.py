import tensorflow.keras as keras
import tensorflow as tf

new_model = tf.keras.models.load_model('epic_num_reader.model')
import numpy as np
from numpy import genfromtxt

x_test = np.array([1,0,0,0,0])
x_test = x_test.reshape(1,5)
print(x_test.shape)
import time
t=time.time()
predictions = new_model.predict(x_test)
print(time.time()-t)
print(np.argmax(predictions[0]))
