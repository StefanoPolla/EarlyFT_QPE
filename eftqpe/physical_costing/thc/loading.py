"""
load THC tensors from file, construct the qubitiztion walk operator bloq and compute lambda.

Adapted from Max's code.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from qualtran.bloqs.qubitization import QubitizationWalkOperator
from qualtran.bloqs.chemistry.thc import PrepareTHC, SelectTHC


def compute_lambda(t_l: NDArray, eta: NDArray, zeta: NDArray) -> float:
    """
    Compute the qubitization 1-norm for the tensor hypercontraction (THC) decomposition.
    """
    SPQ = eta.dot(eta.T)
    cP = np.diag(np.diag(SPQ))
    MPQ_normalized = cP.dot(zeta).dot(cP)  # get normalized zeta in Eq. 11 & 12
    lambda_z = np.sum(np.abs(MPQ_normalized)) * 0.5  # Eq. 13
    lambda_T = np.sum(np.abs(t_l))  # Eq. 19. NOTE: sum over spin orbitals removes 1/2 factor
    lambda_tot = lambda_z + lambda_T  # Eq. 20
    return lambda_tot


def walk_and_lambda_from_file(
    path: str,
    *,
    num_bits_theta: int = 16,
    num_bits_state_prep: int = 10,
    control_values: Tuple | None = [1],
) -> Tuple[QubitizationWalkOperator, float]:
    """
    Load THC tensors from file, construct the qubitiztion walk operator bloq and lambda.

    Parameters:
        path: Path to the THC tensors file.
        num_bits_theta: Number of bits for the angles in the qubitization.
        num_bits_state_prep: Number of bits for the state preparation.
        control_values: List of control values for the controlled walk operator. The default
            is [1], which mirrors the application in quantum phase estimation and QSP. 
           `control_values  = None` will return the uncontrolled walk operator.

    Returns:
        ctrl_walk_operator, lambda_thc
    """
    # load data from THC factorization
    data = np.load(path)

    eri = data["eri"]
    # sanity check: ERI symmetry (Chemists' notation)
    np.testing.assert_allclose(eri, eri.transpose(1, 0, 3, 2), atol=1e-10, rtol=0)
    np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-10, rtol=0)
    np.testing.assert_allclose(eri, eri.transpose(3, 2, 1, 0), atol=1e-10, rtol=0)

    t_l = data["t_l"]  # diagonalized tPQ' (Eq. 18+19)
    # THC tensors
    eta = data["thc_leaf"]  # etaPq
    zeta = data["thc_central"]  # zetaPQ
    # lambda
    lambda_thc = data["lambda_thc"]

    # tensor shapes
    num_mu = eta.shape[0]
    num_spin_orb = 2 * eri.shape[0]

    # re-compute lambda to be sure everything is consistent
    alpha = compute_lambda(t_l, eta, zeta)
    np.testing.assert_allclose(alpha, lambda_thc, atol=1e-10, rtol=0)

    # Build Select and Prepare
    prep_thc = PrepareTHC.from_hamiltonian_coeffs(
        t_l, zeta, num_bits_state_prep=num_bits_state_prep
    )
    sel_thc = SelectTHC(
        num_mu,
        num_spin_orb,
        num_bits_theta=num_bits_theta,
        keep_bitsize=prep_thc.keep_bitsize,
        kr1=16,
        kr2=16,
    )
    walk_operator = QubitizationWalkOperator(select=sel_thc, prepare=prep_thc)

    if control_values is None:
        return walk_operator, lambda_thc
    ctrl_walk_operator = walk_operator.controlled(control_values=control_values)
    return ctrl_walk_operator, lambda_thc
