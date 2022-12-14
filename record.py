from game_control import GameController
from pynput.keyboard import Key, Listener, Controller
from threading import Thread
import time
import subprocess as sp


class GameController():
    def __init__(self) -> None:
        self.listener = Listener(
            on_press=self.handle_press,
            on_release=self.handle_release)
        self.play_thread = Thread(target=self.play)
        self.should_play = False
        self.keyboard = Controller()
        self.pipe = None

    def get_thread(self):
        return self.listener

    def handle_press(self, key):
        if key == Key.end:
            self.should_play = not self.should_play
            if self.should_play:
                command = "ffmpeg -y -video_size 1920x1080 -framerate 60 -f x11grab -i :0.0 -c:v libx264rgb -vf fps=2 -crf 0 -preset ultrafast -color_range 2 output4.mkv"
                self.pipe = sp.Popen(command.split(" "), stdout=sp.PIPE,
                                     stderr=sp.PIPE, bufsize=-1)
                self.play_thread.start()
            else:
                self.pipe.kill()
                self.play_thread.join()
                self.play_thread = Thread(target=self.play)
                print('stopped playing')

    def handle_release(self, key):
        pass

    def play(self):
        if self.should_play:
            print('playing started')

        # while self.should_play:
        #     self.keyboard.press(key='W')
        #     time.sleep(1)
        #     self.keyboard.release(key='W')
        #     time.sleep(0.5)


controller = GameController().get_thread()
controller.start()
controller.join()
