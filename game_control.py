from pynput.keyboard import Key, Listener, Controller
from threading import Thread
import time
import subprocess as sp
from threading import Timer

GAS = 0
BREAK = 1
ENGINE_OFF = 2


class GameController():
    def __init__(self, queue, label, limit) -> None:
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
        self.label = label
        self.limit = limit
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
                self.keyboard.release(Key.space)
                self.keyboard.release('W')
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
            val = self.queue.get()
            self.queue.clear()
            self.keyboard.press('W')
            time.sleep(0.5)
            self.keyboard.release('W')

    def stop(self):
        global BREAK
        while self.state == BREAK:
            val = self.queue.get()
            self.queue.clear()
            self.keyboard.press(Key.space)
            time.sleep(0.5)
            self.keyboard.release(Key.space)

    def play(self):
        if self.should_play:
            print('playing started')

        while self.should_play:
            val = self.queue.get()
            text = "STOP: " + \
                str(val) if val > self.limit else "GO: " + str(val)
            levels = 10
            throttle = max(round(val/levels), round(self.limit/levels))
            spc = int(time.time()) % throttle == 0
            self.label['text'] = str(
                text) + " " + str(spc) + " " + str(int(throttle))
            self.label.pack()
            if val <= self.limit and spc:
                self.state = GAS
                # if not self.for_thread.is_alive() and self.should_play:
                #     self.for_thread = Thread(
                #         target=self.forward)
                #     self.for_thread.start()
                # if self.stop_thread.is_alive():
                #     self.stop_thread.join()
                self.keyboard.release(Key.space)
                self.keyboard.press('W')
            elif not spc:
                self.keyboard.release('W')
                # self.keyboard.release(Key.space)
            elif val > self.limit or spc:
                self.state = BREAK
                # if not self.stop_thread.is_alive() and self.should_play:
                #     self.stop_thread = Thread(target=self.stop)
                #     self.stop_thread.start()
                # if self.for_thread.is_alive():
                #     self.for_thread.join()
                self.keyboard.release('W')
                for i in range(2):
                    self.keyboard.press(Key.space)
                    self.keyboard.release(Key.space)
                self.keyboard.press(Key.space)

            self.queue.clear()

        if self.for_thread.is_alive():
            self.for_thread.join()
        if self.stop_thread.is_alive():
            self.stop_thread.join()
