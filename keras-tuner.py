import tensorflow as tf
from tensorflow.keras import layers
import pickle
import numpy as np
import keras_tuner as kt

# load data
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# X = X/255.0

# data type
X = np.array(X)
y = np.array(y)


def build_model(hp):
    model = tf.keras.Sequential()

    layers.MaxPooling2D(pool_size=(2, 2), input_shape=X.shape[1:]),
    layers.MaxPooling2D(pool_size=(2, 2)),

    for i in range(hp.Int("max_pooling_layers", 1, 5, 1)):
        layers.MaxPooling2D(pool_size=(2, 2)),

    for i in range(hp.Int("conv2d_layers", 1, 5, 1)):
        layers.Conv2D(filters=32, kernel_size=2, activation='relu'),

    layers.Flatten(),

    for i in range(hp.Int("n_layers", 3, 15, 1)):
        model.add(layers.Dense(hp.Int(f"dense_{i}_units", min_value=32, max_value=512, step=32), activation="relu"))
        if hp.Boolean("dropout_layer"):
            model.add(layers.Dropout(hp.Float(f"dropout_{i}_value", min_value=0.0, max_value=0.3, step=0.1)))

    tf.keras.layers.Dense(8)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


tuner = kt.RandomSearch(build_model, objective="val_accuracy", max_trials=50, executions_per_trial=2, directory="logs",
                        overwrite=True, seed=42)
tuner.search(x=X, y=y, epochs=10, validation_split=0.1)

best_model = tuner.get_best_models()[0]

print(best_model.summary())

best_model.save('best-model.h5')
