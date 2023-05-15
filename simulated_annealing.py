from game import TwoPlayerGame

class SimulatedAnnealing:
    def __init__(self, game: TwoPlayerGame, n_epochs: int = 100, initial_temp: float = 100, final_temp: float = 0.01, cooling_rate: float = 0.01):
        self.game = game
        self.n_epochs = n_epochs
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate

    def runSimulation(self):
        pass
