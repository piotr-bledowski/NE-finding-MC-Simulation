import numpy as np
from helpers import normalize

# A simple type of game, since we are only looking for symmetric equilibria in symmetric games
class TwoPlayerSymmetricGame:
    def __init__(self, n: int, payoffs: np.ndarray):
        self.payoffs = np.stack((payoffs, payoffs.T))
        self.strategy = normalize(np.random.rand(n))
        self.pureStrategies = np.identity(n)

    # for readability, I'm indexing players with 1 and 2, hence the -1 below
    def getStrategy(self) -> np.array:
        return self.strategy

    def updateStrategy(self, new_strategy: np.array):
        self.strategy = new_strategy

    def getPayoffs(self) -> np.ndarray:
        return self.payoffs

    # for readability, I'm indexing players with 1 and 2, hence the -1 below
    def getPayoffs(self, player: int) -> np.ndarray:
        return self.payoffs[player-1]
