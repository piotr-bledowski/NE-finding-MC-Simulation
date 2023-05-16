import numpy as np

# Ensuring that an array sums to 1, i.e. is a probability distribution
def normalize(arr: np.array) -> np.array:
    return arr / np.sum(arr)