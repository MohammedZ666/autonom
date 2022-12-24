from game_control import GameController
from game_stream_monodepth2 import GameStreamMono
from game_stream_midas import GameStreamMidas
from utils import Queue
import sys
import tkinter as tk
from tkinter import *

args = sys.argv[1:]
GameStream = None
test = False

if len(args) > 0:
    if 'p' in args:
        preview = True
    else:
        preview = False

    if 'c' in args:
        contrl = True
    else:
        contrl = False

    if 'mono' in args:
        GameStream = GameStreamMono
    else:
        GameStream = GameStreamMidas

    if 't' in args:
        test = True

q = Queue()

if contrl:
    win = tk.Tk()
    label = Label(win, text="PRESS 'END' to activate SAD...", font=('Consolas', '14'),
                  fg='green3',
                  bg='grey19')
    label.pack()
    # Define Window Geometry
    win.overrideredirect(True)
    win.geometry("+5+5")
    win.lift()
    win.wm_attributes("-topmost", True)

limit = 30

streamer = GameStream(q, preview, test, limit).get_thread()
streamer.start()

if contrl:
    controller = GameController(q, label, limit).get_thread()
    controller.start()
    win.mainloop()

streamer.join()
if contrl:
    controller.join()
