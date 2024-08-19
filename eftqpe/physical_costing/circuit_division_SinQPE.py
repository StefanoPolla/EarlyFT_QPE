from typing import Dict, Tuple

import numpy as np
from qualtran.surface_code import MagicCount
from qualtran.surface_code.ccz2t_cost_model import (
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
)

from eftqpe.SinQPE import OptMLESinQPE_params


def opt_circuit_subdivision(epsilon: float, gamma: float) -> Tuple[int, int]:
    """Optimal Sin-QPE circuit subdivision at a fixed precision and error rate

    Args:
        epsilon: target precision on the phase
        gamma: depolarizing error rate per unitary oracle call

    Returns:
        T_max: number of unitary oracle calls within a circuit
        N_rep: number of repetitions of the circuit
    """
    return OptMLESinQPE_params(epsilon, gamma)


def ftqc_physical_cost(
    epsilon: float,
    error_budget: float,
    magic_per_unitary: int | MagicCount,
    n_algo_qubits: int,
    *,
    phys_err: float = 0.001,
    n_factories: int = 1,
) -> Dict:
    """
    Physical cost of a single run of the optimal Sin-QPE circuit with fixed error probability

    Args:
        epsilon: target precision on the phase
        error_budget: error budget for the circuit
        magic_per_unitary: number of Toffoli gates per unitary oracle call or MagicCount
            object with the number of Toffoli and T gates.
        n_algo_qubits: number of logical data qubits (including ancillas)
        phys_err: physical error rate (per pyhsical gate, arXiv:1808.06709)
        n_factories: number of CCZ factories to use in parallel
    """
    T_max = int(np.ceil(np.pi / epsilon))

    tot_magic = T_max * magic_per_unitary
    if not isinstance(tot_magic, MagicCount):
        tot_magic = MagicCount(n_ccz = tot_magic)

    cost, factory, data_block = get_ccz2t_costs_from_grid_search(
        n_magic=tot_magic,
        n_algo_qubits=n_algo_qubits,
        error_budget=error_budget,
        phys_err=phys_err,
        factory_iter=iter_ccz2t_factories(n_factories=n_factories),
        cost_function=(lambda pc: pc.duration_hr * pc.footprint),  # optimize over volume
    )

    return dict(
        physical_cost=cost,
        factory=factory,
        data_block=data_block,
        runtime_hr=cost.duration_hr,
        unitary_oracle_depth=T_max,
    )


def multicircuit_physical_cost(
    epsilon: float,
    gamma: float,
    magic_per_unitary: int | MagicCount,
    n_algo_qubits: int,
    *,
    phys_err: float = 0.001,
    n_factories: int = 1,
) -> Dict:
    """Total physical cost of multi-circuit Sin-QPE algorithm with optimal circuit subdivision

    Args:
        epsilon: target precision on the phase
        gamma: depolarizing error rate per unitary oracle call
        magic_per_unitary: number of Toffoli gates per unitary oracle call or MagicCount
            object with the number of Toffoli and T gates.
        n_algo_qubits: number of logical data qubits (including ancillas)
        phys_err: physical error rate (per pyhsical gate, arXiv:1808.06709)
        n_factories: number of CCZ factories to use in parallel

    Returns:
        dictionary:
            n_repetitions (int): number of distinct circuit runs
            oracle_depth (int): number of unitary oracle calls within a circuit
            physical_cost (PhysicalCost): cost for a single circuit.
            factory (MagicStateFactory): optimal CCZ factory
            data_block (SimpleDataBlock): optimal data block
            t_tot_hr (float): total runtime in hours
            t_ma_hr (float): runtime for a single circuit in hours
            footprint (int): total number of qubits for the algorithm
    """
    T_max, n_repetitions = opt_circuit_subdivision(epsilon, gamma)
    error_budget = 1 - np.exp(-gamma * T_max)

    tot_magic = T_max * magic_per_unitary
    if not isinstance(tot_magic, MagicCount):
        tot_magic = MagicCount(n_ccz = tot_magic)

    cost, factory, data_block = get_ccz2t_costs_from_grid_search(
        n_magic=tot_magic,
        n_algo_qubits=n_algo_qubits,
        error_budget=error_budget,
        phys_err=phys_err,
        factory_iter=iter_ccz2t_factories(n_factories=n_factories),
        cost_function=(lambda pc: pc.duration_hr * pc.footprint),  # optimize over volume
    )
    t_max_hr = cost.duration_hr
    t_tot_hr = cost.duration_hr * n_repetitions

    return dict(
        n_repetitions=n_repetitions,
        oracle_depth=T_max,
        physical_cost=cost,
        factory=factory,
        data_block=data_block,
        t_tot_hr=t_tot_hr,
        t_max_hr=t_max_hr,
        footprint=cost.footprint,
    )
