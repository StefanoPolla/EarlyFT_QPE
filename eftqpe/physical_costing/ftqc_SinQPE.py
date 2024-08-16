import numpy as np

from qualtran.surface_code.ccz2t_cost_model import (
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
)
from qualtran.surface_code import MagicCount


def ftqc_physical_cost(
    epsilon, error_budget, toffoli_per_unitary, n_algo_qubits, *, phys_err=0.001, n_factories=1
):
    """
    Physical cost of a single run of the optimal Sin-QPE circuit with fixed error probability

    Args:
        epsilon (float): target precision on the phase
        error_budget (float): error budget for the circuit
        toffoli_per_unitary (int): number of Toffoli gates per unitary oracle call
        n_algo_qubits (int): number of logical data qubits (including ancillas)
        phys_err (float): physical error rate (per pyhsical gate, arXiv:1808.06709)
        n_factories (int): number of CCZ factories to use in parallel

    Returns:
        dictionary:
            oracle_depth (int): number of unitary oracle calls within a circuit
            physical_cost (PhysicalCost): cost for a single circuit.
            factory (MagicStateFactory): optimal CCZ factory
            data_block (SimpleDataBlock): optimal data block
            runtime_hr (float): total runtime in hours
            footprint (int): total number of qubits for the algorithm
    """
    T_max = int(np.ceil(np.pi / epsilon))
    n_toffoli = T_max * toffoli_per_unitary

    n_magic = MagicCount(n_ccz=n_toffoli)
    cost, factory, data_block = get_ccz2t_costs_from_grid_search(
        n_magic=n_magic,
        n_algo_qubits=n_algo_qubits,
        error_budget=error_budget,
        phys_err=phys_err,
        factory_iter=iter_ccz2t_factories(n_factories=n_factories),
        cost_function=(lambda pc: pc.duration_hr * pc.footprint),  # optimize over volume
    )

    return dict(
        oracle_depth=T_max,
        physical_cost=cost,
        factory=factory,
        data_block=data_block,
        runtime_hr=cost.duration_hr,
        footprint=cost.footprint,
    )
