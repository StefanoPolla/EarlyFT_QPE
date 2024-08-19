import numpy as np
from itertools import product
from tqdm import tqdm
from typing import Sequence, List, Tuple


def circ_dist(phase1, phase2):
    dist = (phase1 - phase2) % (2 * np.pi)
    return np.pi - np.abs(dist - np.pi)


def error_from_probs(probs, ests, phase):
    errors = circ_dist(ests, phase)
    sdv = np.sqrt(np.sum(probs * errors**2))
    return sdv


def fit_prefactor(x, y, power):
    """Least-square fit the prefactor of a power law y = c x^power"""
    return np.exp(np.mean(np.log(y)) - power * np.mean(np.log(x)))


def make_decreasing_function(x: Sequence, y: Sequence) -> Tuple[List, List]:
    """
    Constructs a decreasing function y(x) by sorting x and eliminating points where y_i+1 >= y_1.
    """
    xlist, ylist = zip(*sorted(zip(x, y)))
    current_min = np.inf
    min_x, min_y = [], []
    for x, y in zip(xlist, ylist):
        if y < current_min:
            min_y.append(y)
            min_x.append(x)
            current_min = y
    return min_x, min_y


### Data aggregation


def collect_data(simulate_func, params_list, n_sim):
    """
    Collect data from a simulation function.
    
    TODO: document this function
    """
    # simulate: (phase, *params) -> (error, cost)
    data = []
    for params in tqdm(product(*params_list), total=np.product([len(x) for x in params_list])):
        for _ in range(n_sim):
            true_phase = np.random.uniform(0, 2 * np.pi)
            data.append(simulate_func(true_phase, *params))
    data = np.array(data)
    data = data.reshape(*(len(param) for param in params_list if len(param) != 1), n_sim, 2)

    return data
