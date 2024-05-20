import tensorflow.keras as keras
import tensorflow as tf
from numpy import genfromtxt
print(tf.__version__)
print("hi")
import os
print(os.listdir())

x_train= genfromtxt('Neurosurf\\Aryans\\files_for_training\\bn_tp9_train.csv', delimiter=',')
y_train= genfromtxt('Neurosurf\\Aryans\\files_for_training\\bn_tp9_ans_1.csv', delimiter=',')

x_test = genfromtxt('Neurosurf\\Aryans\\files_for_training\\bn_tp9_test.csv', delimiter=',')
y_test = genfromtxt('Neurosurf\\Aryans\\files_for_training\\bn_tp9_ans_2.csv', delimiter=',')


print(x_train[0])


# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)
model.save('test_model_v1.model')
