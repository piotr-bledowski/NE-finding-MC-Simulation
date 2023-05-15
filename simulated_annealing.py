import numpy as np

from game import TwoPlayerGame

class SimulatedAnnealing:
    def __init__(self, game: TwoPlayerGame, n_epochs: int = 100, initial_temp: float = 100, final_temp: float = 0.01, cooling_rate: float = 0.01):
        self.game = game
        self.n_epochs = n_epochs
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate

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