import numpy as np

class TwoPlayerGame:
    def __init__(self, payoffs1: np.ndarray, payoffs2: np.ndarray, strategy1: np.array, strategy2: np.array):
        self.payoffs = np.stack((payoffs1, payoffs2))
        self.strategies = np.vstack((strategy1, strategy2))

    # for readability, I'm indexing players with 1 and 2, hence the -1 below
    def getStrategy(self, player: int) -> np.array:
        return self.strategies[player-1]

    def updateStrategy(self, player: int, new_strategy: np.array):
        self.strategies[player-1] = new_strategy

    def getPayoffs(self) -> np.ndarray:
        return self.payoffs

    # for readability, I'm indexing players with 1 and 2, hence the -1 below
    def getPayoffs(self, player: int) -> np.ndarray:
        return self.payoffs[player-1]

    def setStrategies(self, strategy1: np.array, strategy2: np.array):
        self.strategies = np.vstack((strategy1, strategy2))

    def expectedPayoffs(self, player: int) -> np.array:
        # multiply player's payoffs by respective probability of the opponent playing that action
        # then sum the rows so that each action of the player has an expected payoff

        p0 = self.payoffs[0]
        p1 = np.transpose(self.payoffs[1])

        mp0 = self.payoffs[0] * self.strategies[1]
        mp1 = np.transpose(self.payoffs[1]) * self.strategies[0]

        return np.sum(self.payoffs[0] * self.strategies[1], axis=1) if player == 1\
            else np.sum(np.transpose(self.payoffs[1]) * self.strategies[0], axis=1)
