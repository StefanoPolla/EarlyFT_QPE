import numpy as np
from itertools import product
from tqdm import tqdm
from typing import Sequence, List, Tuple


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