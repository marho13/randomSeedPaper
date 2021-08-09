import random
import gym

class environment:
    def __init__(self):
        self.env = gym.make("CarRacing-v0")

    def run(self, seed):
        self.env.seed(seed)
        random.seed(seed)
        self.env.reset()

    def start(self):
        self.env.reset()

e = environment()
e.run(43)
e.start()
# e.run(43)

# Track generation: 1208..1514 -> 306-tiles track
# Track generation: 1203..1508 -> 305-tiles track 42

# Track generation: 1208..1514 -> 306-tiles track 42
# Track generation: 1121..1405 -> 284-tiles track 43

# Track generation: 1121..1405 -> 284-tiles track 43
# Track generation: 1121..1405 -> 284-tiles track 43

# Track generation: 1121..1405 -> 284-tiles track
# Track generation: 1170..1467 -> 297-tiles track

