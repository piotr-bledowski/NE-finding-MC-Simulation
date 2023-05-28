from simulated_annealing import SimulatedAnnealing
from game import TwoPlayerSymmetricGame
import numpy as np
import matplotlib.pyplot as plt
from helpers import normalize, cost, payoff

# Example of multiple global minima
#payoffs = np.array([[1, 2, 3], [3, 5, 1], [4, 0, 2]]) * 1000

payoffs = normalize(np.array([[90, 18, 89], [95, 52, 44], [44, 12, 95]]))
#payoffs = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
print(payoffs)

game = TwoPlayerSymmetricGame(3, payoffs)

simulation = SimulatedAnnealing(game, n_epochs=50, initial_temp=50, final_temp=0.01, cooling_rate=0.01, acceptance_treshold='sigmoid', step_size=0.01)

simulation.runSimulation()

print(simulation.getResult())
print(simulation.error)

# fig, ax = plt.subplots()
#
# X = np.linspace(0, 1, 100)
# Y = np.linspace(0, 1, 100)
#
# Z = np.array([[cost(game.payoffs[0], game.pureStrategies, np.array([x, y, 1-x-y])) if x + y < 1 else 0 for x in X] for y in Y])
#
# ax.contourf(X, Y, Z, levels=np.linspace(0, Z.max(), 1000))
