import numpy as np

# Ensuring that an array sums to 1, i.e. is a valid probability distribution
def normalize(arr: np.array) -> np.array:
    return arr / np.sum(arr)

def cost(payoffs: np.ndarray, pure_strategies: np.ndarray, strat: np.array) -> float:
    return sum(max([0, payoff(payoffs, p, strat) - payoff(payoffs, strat, strat)])**2 for p in pure_strategies)

def payoff(payoffs: np.ndarray, strategy1: np.array, strategy2: np.array) -> float:
    po = payoffs * strategy1[:, None] * strategy2 # Multiplying payoff matrix vertically by strategy1, then horizontally by strategy2
    return po.sum()