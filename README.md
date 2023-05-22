# Simulated-Annealing-Symmetric-Nash-Equilibrium-Search

The goal of this project is to approach the problem of finding a symmetric Nash Equilibrium in a symmetric two-player normal form game as a function minimization problem.

## Algorithm
The algorithm used is simulated annealing.
The motivation behind choosing this particular method is that the objective function does not guarantee all minima to be global, i.e. equal 0 as that is the game-theoretic interpretation here.
Hence, stochastic approach is used to find the global minimum of the objective function which is the symmetric Nash Equilibrium of the given game.
