from game_control import GameController
from game_stream import GameStream
from queue import Queue

q = Queue()

#controller = GameController(q).get_thread()
streamer = GameStream(q).get_thread()

streamer.start()
# controller.start()

streamer.join()
# controller.join()
