from simulated_annealing import SimulatedAnnealing
from game import TwoPlayerSymmetricGame
import numpy as np
from helpers import normalize
import matplotlib.pyplot as plt
from helpers import cost, payoff

# po1 = np.array([[2, 77, 48], [42, 73, 26], [67, 86, 11]])
# po2 = np.array([[2, 42, 67], [77, 73, 86], [48, 26, 11]])
payoffs = np.array([[1, 2, 3], [3, 5, 1], [4, 0, 2]]) * 1000

game = TwoPlayerSymmetricGame(3, payoffs)

simulation = SimulatedAnnealing(game, n_epochs=100, initial_temp=50, final_temp=0.01, cooling_rate=0.01, step_size=0.001, step='normal')

simulation.runSimulation()

print(simulation.getResult())
print(simulation.error[-1])


# fig, ax = plt.subplots()
#
# X = np.linspace(0, 1, 100)
# Y = np.linspace(0, 1, 100)
#
# Z = np.array([[cost(game.payoffs[0], game.pureStrategies, np.array([x, y, 1-x-y])) for x in X] for y in Y])
#
# ax.contourf(X, Y, Z, levels=np.linspace(0, Z.max(), 1000))
#
# plt.show()













#
# simulation = SimulatedAnnealing(game, n_epochs=50, initial_temp=5, final_temp=0.01, cooling_rate=0.01, step_size=0.0001, step='normal')
#
# simulation.runSimulation()
#
# print(simulation.getResult())
# print(cost(simulation.game))

# fig, ax = plt.subplots(nrows=1, ncols=3)
#
# p = np.linspace(0, 1, 100)
# q = np.linspace(0, 1, 100)
#
# X, Y = np.meshgrid(p, q)
#
#
# Z1 = np.array([[np.var(np.array([po2[0][k]*i + po2[1][k]*j + po2[2][k]*(1-i-j) for k in range(3)]))
#                 for i in p] for j in q])
#
# Z2 = np.array([[np.var(np.array([po1[k][0]*i + po1[k][1]*j + po1[k][2]*(1-i-j) for k in range(3)]))
#                 for i in p] for j in q])
#
# levels = np.linspace(0, Z1.max(), 100)
#
# ax[0].contourf(X, Y, Z1, levels=levels)
# ax[1].contourf(X, Y, Z2, levels=levels)
# ax[2].plot(np.linspace(0, len(simulation.error), len(simulation.error)), np.array(simulation.error))
#
# #plt.plot(np.arange(len(simulation.error)), np.array(simulation.error))
# plt.show()
