from simulated_annealing import SimulatedAnnealing
from game import TwoPlayerGame
import numpy as np
from helpers import normalize
import matplotlib.pyplot as plt
from helpers import cost

as1 = np.array([-1, 1])
as2 = np.array([-1, 1])
po1 = np.vstack((np.array([3, 1]), np.array([0, 2])))
po2 = np.vstack((np.array([2, 1]), np.array([0, 3])))
s1 = normalize(np.random.rand(2))
s2 = normalize(np.random.rand(2))

print(np.vstack((s1, s2)))

game = TwoPlayerGame(as1, as2, po1, po2, s1, s2)

simulation = SimulatedAnnealing(game, n_epochs=100, initial_temp=100, final_temp=0.1, cooling_rate=0.1)

simulation.runSimulation()

print(simulation.getResult())
print(cost(simulation.game))

fig, ax = plt.subplots()

p = np.linspace(0, 1, 100)
q = np.linspace(0, 1, 100)

X, Y = np.meshgrid(p, q)

Z = np.array([[np.std(np.array([po1[0][0]*i + po1[0][1]*(1-i), po1[1][0]*i + po1[1][1]*(1-i)]))
     + np.std(np.array([po2[0][0]*j + po2[0][1]*(1-j), po2[1][0]*j + po2[1][1]*(1-j)])) for i in p] for j in q])

levels = np.linspace(0, Z.max(), 100)

ax.contourf(X, Y, Z, levels=levels)

#plt.plot(np.arange(len(simulation.error)), np.array(simulation.error))
plt.show()
