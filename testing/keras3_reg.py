import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

print("Keras version:", keras.__version__)

X = tf.ones((10, 5))
y = tf.ones((10, 1))

model = keras.Sequential([
    keras.layers.InputLayer(shape=(5,)),
    keras.layers.Dense(1, kernel_initializer="ones", bias_initializer="zeros"),
    keras.layers.ActivityRegularization(l2=0.001)
])
model.compile(loss="mse")

print("Loss:", model.evaluate(X, y, verbose=0))
