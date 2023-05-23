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

    # def expectedPayoffs(self, player: int) -> np.array:
    #     # multiply player's payoffs by respective probability of the opponent playing that action
    #     # then sum the rows so that each action of the player has an expected payoff
    #
    #     p0 = self.payoffs[0]
    #     p1 = np.transpose(self.payoffs[1])
    #
    #     mp0 = self.payoffs[0] * self.strategies[1]
    #     mp1 = np.transpose(self.payoffs[1]) * self.strategies[0]
    #
    #     return np.sum(self.payoffs[0] * self.strategies[1], axis=1) if player == 1\
    #         else np.sum(np.transpose(self.payoffs[1]) * self.strategies[0], axis=1)
