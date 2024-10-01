from typing import List

import numpy as np
import pandas as pd
from qualtran.resource_counting import QubitCount, get_cost_value
from tqdm import tqdm

from .physical_costing import multicircuit_physical_cost, thc, ftqc_physical_cost


def estimate_thc_resources(thc_file: str, delta_e: float = 1e-3, n_factories: int = 1, gammas: float | List | np.ndarray = 1e-6) -> pd.DataFrame:
    # construct the walk operator
    ctrl_walk, lambda_thc = thc.walk_and_lambda_from_file(thc_file)

    # call graph
    g, sigma = thc.walk_call_graph(ctrl_walk)
    print(*sorted(f"{str(k):30s}: {v}" for k, v in sigma.items()), sep="\n")
    
    # count logical resources
    magic_per_walk = thc.magic_from_sigma(sigma)
    print(magic_per_walk)
    
    # count the number of qubits
    total_qubits = get_cost_value(ctrl_walk, QubitCount())
    print(total_qubits)

    if isinstance(gammas, float):
        gammas = [gammas]

    data = []
    for gamma in tqdm(gammas):
        cost = multicircuit_physical_cost(
            epsilon=delta_e / lambda_thc,
            gamma=gamma,
            magic_per_unitary=magic_per_walk,
            n_algo_qubits=total_qubits,
            n_factories=n_factories,
        )
        cost["gamma"] = gamma
        cost["n_physical_qubits"] = cost["physical_cost"].footprint
        data.append(cost)
    df = pd.DataFrame(data)
    df['delta_e'] = delta_e
    df['n_factories'] = n_factories
    df['lambda_thc'] = lambda_thc
    df['total_qubits'] = total_qubits
    df['magic_per_walk'] = magic_per_walk
    return df


def thc_ftqc_physical_cost(ccz_count, total_qubits, lambda_thc, error_budget=0.01, delta_e=1e-3, n_factories=1):
    """
    Wrapper of ftqc_physical_cost for THC with standard parameters.
    """
    toffoli_per_unitary = ccz_count # TODO change to general magic count

    cost = ftqc_physical_cost(
        epsilon=delta_e / lambda_thc,
        error_budget=error_budget,
        toffoli_per_unitary=toffoli_per_unitary,
        n_algo_qubits=total_qubits,
        n_factories=n_factories,
    )
    return cost
