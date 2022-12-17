from game_control import GameController
from game_stream_monodepth2 import GameStreamMono
from game_stream_midas import GameStreamMidas

from queue import Queue
import sys


args = sys.argv[1:]
GameStream = None

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


q = Queue()

if contrl:
    controller = GameController(q).get_thread()
    controller.start()

streamer = GameStream(q, preview).get_thread()
streamer.start()
streamer.join()
if contrl:
    controller.join()
