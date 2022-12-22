from pynput.keyboard import Key, Listener, Controller
from threading import Thread
import time
import subprocess as sp
from threading import Timer

GAS = 0
BREAK = 1
ENGINE_OFF = 2


class GameController():
    def __init__(self, queue) -> None:
        self.listener = Listener(
            on_press=self.handle_press,
            on_release=self.handle_release)
        self.queue = queue
        self.play_thread = Thread(target=self.play)
        self.should_play = False
        self.keyboard = Controller()
        self.for_thread = Thread(target=self.forward)
        self.stop_thread = Thread(target=self.stop)
        self.state = GAS
        self.pipe = None

    def get_thread(self):
        return self.listener

    def handle_press(self, key):
        if key == Key.end:
            self.should_play = not self.should_play
            if self.should_play:
                # command = "ffmpeg -video_size 1920x1080 -framerate 15 -f x11grab -i :0.0 -c:v libx264rgb -crf 0 -preset ultrafast -color_range 2 output.mkv"
                # self.pipe = sp.Popen(command.split(" "), stdout=sp.PIPE,
                #                      stderr=sp.PIPE, bufsize=-1)
                self.play_thread.start()
            else:
                # self.pipe.kill()
                self.state = ENGINE_OFF
                self.play_thread.join()
                self.play_thread = Thread(target=self.play)
                self.for_thread = Thread(target=self.forward)
                self.stop_thread = Thread(target=self.stop)
                print('stopped playing')

    def handle_release(self, key):
        pass

    def forward(self):
        global GAS
        while self.state == GAS:
            dis = self.queue.get()
            self.queue.clear()

            if dis < 30:
                self.keyboard.press('W')
                time.sleep(0.5)
                self.keyboard.release('W')

            else:
                self.keyboard.press('W')
                time.sleep(0.25)
                self.keyboard.release('W')
        # print("Forward loop ended")

    def stop(self):
        while self.state == BREAK:
            dis = self.queue.get()
            self.queue.clear()

            # print('breaking', dis)
            self.keyboard.press(Key.space)
            time.sleep(0.01)
            self.keyboard.release(Key.space)
        # print("Stop loop ended")

    def play(self):
        if self.should_play:
            print('playing started')

        while self.should_play:
            dis = self.queue.get()
            self.queue.clear()

            if dis <= 30:
                self.state = GAS
                if not self.for_thread.is_alive() and self.should_play:
                    self.for_thread = Thread(
                        target=self.forward)
                    self.for_thread.start()
                if self.stop_thread.is_alive():
                    self.stop_thread.join()

            else:
                self.state = BREAK
                if not self.stop_thread.is_alive() and self.should_play:
                    self.stop_thread = Thread(target=self.stop)
                    self.stop_thread.start()
                if self.for_thread.is_alive():
                    self.for_thread.join()

        if self.for_thread.is_alive():
            self.for_thread.join()
        if self.stop_thread.is_alive():
            self.stop_thread.join()
