import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers


data_augmentation = tf.keras.Sequential([
  layers.InputLayer(input_shape=(224, 224, 3)),
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.5, 0.2)
])

def create_model():
    model = tf.keras.Sequential([
        data_augmentation,
        layers.Conv2D(10, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(15, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(30, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics='accuracy'
    )

    return model

model = create_model()
model.load_weights('weights')

def show_pred(pred):
    return 'Santa' if pred == 1 else 'Not Santa'

def add_understanding(perc):
    return "So it's not santa" if perc < 50 else "So it's most likely santa"

def pred(im_path):
    img = tf.keras.utils.load_img(im_path)
    img = tf.keras.utils.img_to_array(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img = tf.reshape(img, (1, 224, 224, 3))
    return f"Model is {str(model.predict(img)[0][1] * 100)[:4]}% sure it's santa \n{add_understanding(model.predict(img)[0][1] * 100)}. \nHappy new Year!"