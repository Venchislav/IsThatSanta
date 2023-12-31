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



__author__ = 'Venchislav_code'

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.core.window import Window

import cv2


class IsThatSantaApp(App):

    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        self.model = create_model()
        self.model.load_weights('weights')

        # Camera image
        self.img1 = Image()
        layout = BoxLayout()
        layout.orientation = 'vertical'
        layout.add_widget(self.img1)

        # Prediction text
        self.text = Label()
        self.text.text = ''
        self.text.color = 'black'
        self.text.center_x = 0.5
        layout.add_widget(self.text)

        # Photo button
        button = Image(source='resources/Photo_btn.png')
        button.bind(on_touch_down=self.shoot)
        button.size_hint = (.5, .5)
        button.pos_hint = {'x': 0.25}
        layout.add_widget(button)

        self.capture = cv2.VideoCapture(0)
        cv2.namedWindow("CV2 Is That Santa")
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout
    
    def shoot(self, *args):
        img_name = "Images_taken\opencv_frame_{}.png".format('0')
        cv2.imwrite(img_name, self.frame)
        print("{} written!".format(img_name))
        self.predict(img_name)

    def predict(self, img):
        self.text.text = pred(img)

    def update(self, dt):
        ret, self.frame = self.capture.read()
        cv2.imshow("CV2 Is That Santa", self.frame)
        buf1 = cv2.flip(self.frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        
        self.img1.texture = texture1


if __name__ == '__main__':
    IsThatSantaApp().run()
    cv2.destroyAllWindows()
