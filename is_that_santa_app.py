__author__ = 'Venchislav_code'

from data_is_that_santa import create_model, pred
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

        self.img1 = Image()
        """self.img1.size_hint = (.5, .5)
        self.img1.pos_hint = {'x': .5, 'y': .5}"""
        layout = BoxLayout()
        layout.orientation = 'vertical'
        layout.add_widget(self.img1)

        self.text = Label()
        self.text.text = ''
        self.text.color = 'black'
        layout.add_widget(self.text)

        button = Button(text='Is That Santa?')
        """button.size_hint = (.5, .5)
        button.pos_hint = {'x': -.5, 'y': .0}"""
        button.bind(on_press=self.shoot)
        button.background_color = '#34A65F'
        button.color = 'white'
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
