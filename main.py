from simulated_annealing import SimulatedAnnealing
from game import TwoPlayerGame
import numpy as np
from helpers import normalize
import matplotlib.pyplot as plt

as1 = np.array([-1, 1])
as2 = np.array([-1, 1])
po1 = np.vstack((np.array([2, 0]), np.array([0, 1])))
po2 = np.vstack((np.array([1, 0]), np.array([0, 2])))
s1 = normalize(np.random.rand(2))
s2 = normalize(np.random.rand(2))

print(np.vstack((s1, s2)))

game = TwoPlayerGame(as1, as2, po1, po2, s1, s2)

simulation = SimulatedAnnealing(game, n_epochs=100, initial_temp=0.01, final_temp=0.000001, cooling_rate=0.000001)

simulation.runSimulation()

print(simulation.getResult())

plt.plot(np.arange(len(simulation.error)), np.array(simulation.error))
plt.show()
