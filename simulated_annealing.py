import numpy as np

from game import TwoPlayerSymmetricGame
from helpers import normalize, cost

class SimulatedAnnealing:
    def __init__(self,
                 game: TwoPlayerSymmetricGame,
                 n_epochs: int = 100,
                 initial_temp: float = 100,
                 final_temp: float = 0.01,
                 cooling_rate: float = 0.01,
                 temp_reduction: str = 'linear', # either linear or geometric
                 acceptance_treshold: str = 'sigmoid', # either sigmoid or exp
                 step: str = 'fixed',
                 step_size: float = 0.001):

        self.game = game
        self.n_epochs = n_epochs
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.temp_reduction = temp_reduction
        self.acceptance_treshold = acceptance_treshold
        self.step = step
        self.step_size = step_size
        self.error = None

    def runSimulation(self):
        temp = self.initial_temp

        while temp >= self.final_temp:
            for _ in range(self.n_epochs):
                current_cost = cost(self.game.payoffs[0], self.game.pureStrategies, self.game.strategy)
                self.error = current_cost
                # take a step, i.e. change one of the probabilities in one of the players' strategy
                next_strategy = self.nextStep()
                next_cost = cost(self.game.payoffs[0], self.game.pureStrategies, next_strategy)
                delta = current_cost - next_cost
                # always accept better states
                if delta > 0:
                    self.game.updateStrategy(next_strategy)
                # if the new state is worse, accept it with probability derived from current temperature and change in cost
                else:
                    if self.acceptance_treshold == 'sigmoid':
                        if 1/(1+np.exp(delta / temp)) > np.random.rand():
                            self.game.updateStrategy(next_strategy)
                    elif self.acceptance_treshold == 'exp':
                        if np.exp(delta / temp) > np.random.rand():
                            self.game.updateStrategy(next_strategy)
            if self.temp_reduction == 'linear':
                temp -= self.cooling_rate
            elif self.temp_reduction == 'geometric':
                temp *= self.cooling_rate

    # Three possible approaches here:
    # 1. taking a step of a fixed size
    # 2. taking a step from a uniform distribution
    # 3. taking a step from a normal distribution (Gaussian random walk)
    def nextStep(self) -> np.array:
        strategy = self.game.strategy.copy()
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
        # in case a probability goes below 0
        if strategy[action] < 0:
            return self.game.strategy
        return strategy

    # return the Nash Equilibrium approximation found
    def getResult(self) -> np.ndarray:
        return self.game.strategy
