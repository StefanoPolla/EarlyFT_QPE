from qualtran.bloqs.chemistry.hubbard_model.qubitization import get_walk_operator_for_hubbard_model
from eftqpe.physical_costing import multicircuit_physical_cost, ftqc_physical_cost
import numpy as np
from typing import Tuple


def fermi_hubbard_walk_costs(
    side: int, nbits_for_rotations: int = 10, t: float = 1, mu: float = 4
) -> Tuple[int, int, int]:
    """
    Calculate the logical costs of the qubitization walk operator for the Fermi-Hubbard model.

    Args:
        side: The side length of the square lattice.
        nbits_for_rotations: The number of bits used for the phase gradient rotations.
        t: The hopping parameter.
        mu: The chemical potential.

    Returns:
        ccz_count: The number of CCZ gates required to implement the walk operator. TODO: change to general magic count
        total_qubits: total logical qubits (spin orbitals + auxiliary qubits).
        qlambda: qubitization 1-norm.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
    """
    x_dim, y_dim = int(side), int(side)
    t = 1
    mu = 4 * t
    n_spin_orbitals = x_dim * y_dim * 2
    qlambda = 2 * n_spin_orbitals * t + (n_spin_orbitals * mu) // 2  # eq. 60

    walk = get_walk_operator_for_hubbard_model(x_dim, y_dim, t, mu).controlled(1)
    complexity = walk.t_complexity()

    ccz_from_rotations = (
        complexity.rotations * nbits_for_rotations
    )  # CCZ cost for phase gradient rotations
    ccz_from_logic = complexity.t / 4  # in this circuit all T gates come from CCZ gates

    # TODO figure out a way to directly extract CCZ costs from qualtran
    ccz_count = ccz_from_rotations + ccz_from_logic

    spin_orbitals = x_dim * y_dim * 2
    ancilla_count = np.ceil(12 + 3 * np.log2(spin_orbitals))  # eq. 64
    ancilla_count += nbits_for_rotations  # phase gradient rotation
    ancilla_count += 1  # for the controlled phase gradient rotation
    total_qubits = spin_orbitals + ancilla_count

    return ccz_count, total_qubits, qlambda


def fermi_hubbard_multicircuit_physical_cost(side, gamma, delta_e=0.01, n_factories=1):
    """
    Wrapper of multicircuit_physical_cost for the square fermi-hubbard model with standard parameters.
    """
    ccz_count, total_qubits, qlambda = fermi_hubbard_walk_costs(side)
    toffoli_per_unitary = ccz_count  # TODO change to general magic count

    cost = multicircuit_physical_cost(
        epsilon=delta_e / qlambda,
        gamma=gamma,
        toffoli_per_unitary=toffoli_per_unitary,
        n_algo_qubits=total_qubits,
        n_factories=n_factories,
    )

    return cost

def fermi_hubbard_ftqc_physical_cost(side, error_budget = 0.01, delta_e = 0.01, n_factories=1):
    """
    Wrapper of ftqc_physical_cost for the square fermi-hubbard model with standard parameters.
    """
    ccz_count, total_qubits, qlambda = fermi_hubbard_walk_costs(side)
    toffoli_per_unitary = ccz_count # TODO change to general magic count

    cost = ftqc_physical_cost(
        epsilon=delta_e / qlambda,
        error_budget=error_budget,
        toffoli_per_unitary=toffoli_per_unitary,
        n_algo_qubits=total_qubits,
        n_factories=n_factories,
    )
    return cost