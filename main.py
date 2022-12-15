from game_control import GameController
from game_stream_midas import GameStream
from queue import Queue
import sys


args = sys.argv[1:]
if len(args) > 0 and args[0] == 'p':
    preview = True
else:
    preview = False


if len(args) > 1 and args[1] == 'c':
    contrl = True
else:
    contrl = False


q = Queue()

if contrl:
    controller = GameController(q).get_thread()
    controller.start()
    controller.join()

streamer = GameStream(q, preview).get_thread()
streamer.start()
streamer.join()
