import numpy as np

from game import TwoPlayerGame
from helpers import normalize

class SimulatedAnnealing:
    def __init__(self, game: TwoPlayerGame, n_epochs: int = 100, initial_temp: float = 100, final_temp: float = 0.01, cooling_rate: float = 0.01, step: str = 'fixed', step_size: float = 0.01):
        self.game = game
        self.n_epochs = n_epochs
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.step = step
        self.step_size = step_size

    def runSimulation(self):
        temp = self.initial_temp

        while temp >= self.final_temp:
            for _ in range(self.n_epochs):
                current_cost = self.cost()

    # This can either be standard deviation or variance of opponent's expected payoffs
    # as Nash Equilibrium occurs when said SD or variance is equal to 0
    def cost(self) -> float:
        expected_payoffs1 = self.game.expectedPayoffs(1)
        expected_payoffs2 = self.game.expectedPayoffs(2)
        return np.var(expected_payoffs1) + np.var(expected_payoffs2)

    # Three possible approaches here
    # 1. taking a step of a fixed size
    # 2. taking a step from a uniform distribution (regular random walk)
    # 3. taking a step from a normal distribution (Gaussian random walk)
    def nextStep(self):
        # randomly choosing a player
        player = np.random.randint(2) + 1
        strategy = self.game.getStrategy(player)
        # randomly choose an action
        action = np.random.randint(strategy.length)
        if self.step == 'fixed':
            strategy[action] += np.random.choice([-self.step_size, self.step_size])
        elif self.step == 'uniform':
            # here self.step_size serves as a max possible step size
            strategy[action] += np.random.rand(-self.step_size, self.step_size)
        elif self.step == 'normal':
            strategy[action] = np.random.normal(loc=strategy[action], scale=self.step_size)
        strategy = normalize(strategy)
        self.game.updateStrategy(player, strategy)
