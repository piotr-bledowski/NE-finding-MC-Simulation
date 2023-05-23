import numpy as np
from game import TwoPlayerGame

# Ensuring that an array sums to 1, i.e. is a probability distribution
def normalize(arr: np.array) -> np.array:
    return arr / np.sum(arr)

# This can either be standard deviation or variance of opponent's expected payoffs
# as Nash Equilibrium occurs when said SD or variance is equal to 0
# THIS IS NOT TRUE AT ALL, IGNORE THIS
# def cost(game: TwoPlayerGame) -> float:
#     expected_payoffs1 = game.expectedPayoffs(1)
#     expected_payoffs2 = game.expectedPayoffs(2)
#     var1 = np.var(expected_payoffs1)
#     var2 = np.var(expected_payoffs2)
#
#     return np.var(expected_payoffs1) + np.var(expected_payoffs2)


def cost(game: TwoPlayerGame, strat: np.array) -> float:
    return sum(max([0, payoff(game, p, strat) - payoff(game, strat, strat)])**2 for p in game.pureStrategies)


def payoff(game: TwoPlayerGame, strategy1: np.array, strategy2: np.array) -> float:
    payoffs = game.getPayoffs(1) * strategy1 * strategy2.T
    return payoffs.sum()