import numpy as np

from game import TwoPlayerSymmetricGame
from helpers import normalize, cost
from copy import deepcopy

class SimulatedAnnealing:
    def __init__(self, game: TwoPlayerSymmetricGame, n_epochs: int = 100, initial_temp: float = 100, final_temp: float = 0.01, cooling_rate: float = 0.01, step: str = 'fixed', step_size: float = 0.001):
        self.game = game
        self.n_epochs = n_epochs
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.step = step
        self.step_size = step_size
        self.error = []

    def runSimulation(self):
        temp = self.initial_temp

        while temp >= self.final_temp:
            for _ in range(self.n_epochs):
                current_cost = cost(self.game)
                self.error.append(current_cost)
                # take a step, i.e. change one of the probabilities in one of the players' strategy
                next_state = self.nextStep()
                next_cost = cost(next_state)
                delta = current_cost - next_cost
                # always accept better states
                if delta > 0:
                    self.game = next_state
                # if the new state is worse, accept it with probability derived from current temperature and change in cost
                elif 1/(1+np.exp(-delta/temp)) > np.random.rand(): # sigmoid
                    self.game = next_state
            temp -= self.cooling_rate

    # Three possible approaches here
    # 1. taking a step of a fixed size
    # 2. taking a step from a uniform distribution (regular random walk)
    # 3. taking a step from a normal distribution (Gaussian random walk)
    def nextStep(self) -> TwoPlayerSymmetricGame:
        new_game = deepcopy(self.game)
        # randomly choosing a player
        player = np.random.randint(2) + 1
        strategy = new_game.getStrategy(player)
        # randomly choose an action
        action = np.random.randint(strategy.size)
        if self.step == 'fixed':
            strategy[action] += np.random.choice([-self.step_size, self.step_size])
        elif self.step == 'uniform':
            # here self.step_size serves as a max possible step size
            strategy[action] += np.random.uniform(-self.step_size, self.step_size)
        elif self.step == 'normal':
            strategy[action] = np.random.normal(loc=strategy[action], scale=self.step_size)
        strategy = normalize(strategy)
        new_game.updateStrategy(strategy)
        # in case a probability goes below 0
        if strategy[action] < 0:
            return self.game
        return new_game

    # return the Nash Equilibrium approximation found
    def getResult(self) -> np.ndarray:
        return np.vstack((self.game.getStrategy(1), self.game.getStrategy(2)))
