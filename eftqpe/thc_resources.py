from typing import List
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from qualtran.resource_counting import QubitCount, get_cost_value
from tqdm import tqdm
from openfermion.resource_estimates import thc as thc_of

from .physical_costing import multicircuit_physical_cost, thc, ftqc_physical_cost


try:
    import pybtas
    have_pybtas = True
except ImportError:
    have_pybtas = False

thc_deps_available = thc_of.HAVE_DEPS_FOR_RESOURCE_ESTIMATES and have_pybtas


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


@dataclass
class THCFactorizationResult:
    h1: NDArray
    eri: NDArray
    core_energy: float
    #
    eri_to_factorize: NDArray
    eri_reconstructed: NDArray
    tpq_prime: NDArray
    #
    t_l: NDArray
    thc_leaf: NDArray
    thc_central: NDArray
    #
    error_reconstruction: float
    lambda_thc: float
    a1: float = 0.0
    a2: float = 0.0
    nalpha: int = None
    nbeta: int = None
    error_ccsd: float = None

    def to_npz(self, path: str):
        np.savez(path, **asdict(self))

    @staticmethod
    def from_npz(path: str):
        data = np.load(path)
        return THCFactorizationResult(**data)


def shift_eri(eri: NDArray) -> NDArray:
    eri_to_factorize = eri.copy()
    diag_pr = np.einsum("pprr->pr", eri_to_factorize)
    a2 = np.median(diag_pr)
    diag_pr -= a2
    return eri_to_factorize, a2


def compute_lambda_thc(t_l: NDArray, eta: NDArray, zeta: NDArray) -> float:
    SPQ = eta.dot(eta.T)
    cP = np.diag(np.diag(SPQ))
    MPQ_normalized = cP.dot(zeta).dot(cP)  # get normalized zeta in Eq. 11 & 12
    lambda_z = np.sum(np.abs(MPQ_normalized)) * 0.5  # Eq. 13
    lambda_T = np.sum(np.abs(t_l))  # Eq. 19. NOTE: sum over spin orbitals removes 1/2 factor
    lambda_tot = lambda_z + lambda_T  # Eq. 20
    return lambda_tot


def run_thc(
    h1: NDArray,
    eri: NDArray,
    nthc: int,
    symm_shift: bool = True,
    core_energy: float = 0.0,
    **kwargs
) -> THCFactorizationResult:
    if not thc_deps_available:
        raise ImportError("THC dependencies are not available. Please install pybtas and jax.")
    np.testing.assert_allclose(eri, eri.transpose(0, 1, 3, 2), atol=1e-10, rtol=0)
    np.testing.assert_allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-10, rtol=0)
    np.testing.assert_allclose(eri, eri.transpose(1, 0, 3, 2), atol=1e-10, rtol=0)
    np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-10, rtol=0)
    np.testing.assert_allclose(eri, eri.transpose(3, 2, 1, 0), atol=1e-10, rtol=0)
    eri_to_factorize = eri.copy()
    a1 = a2 = 0.0
    if symm_shift:
        eri_to_factorize, a2 = shift_eri(eri_to_factorize)
    eri_rr, thc_leaf, thc_central, _ = thc_of.factorize(eri_to_factorize, nthc=nthc, **kwargs)
    l2_error = np.linalg.norm(eri_rr - eri_to_factorize)
    tpq_prime = (
        h1
        - 0.5 * np.einsum("illj->ij", eri, optimize=True)
        + np.einsum("llij->ij", eri, optimize=True)
    )
    t_l = np.linalg.eigvalsh(tpq_prime)
    if symm_shift:
        a1 = np.median(t_l)
        t_l -= a1
    lambda_thc = compute_lambda_thc(t_l, thc_leaf, thc_central)
    ret = THCFactorizationResult(
        h1=h1,
        eri=eri,
        core_energy=core_energy,
        #
        eri_to_factorize=eri_to_factorize,
        eri_reconstructed=eri_rr,
        tpq_prime=tpq_prime,
        #
        t_l=t_l,
        thc_leaf=thc_leaf,
        thc_central=thc_central,
        #
        error_reconstruction=l2_error,
        lambda_thc=lambda_thc,
        #
        a1=a1,
        a2=a2,
    )
    return ret
