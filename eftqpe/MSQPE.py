from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from eftqpe.SinQPE import SinQPE_FI as Fisher_information
from eftqpe.SinQPE import SinQPE_Holevo_error as Holevo_error

### Algorithm constants for threshold errors calculation

# constant c1 such that T1 = floor(c1/gamma^(1/3)) is the maximal depth for which 1 sample is used
DEPTH_FACTOR_1 = (2 / 3 * np.pi**2) ** (1 / 3)

# constant c2 such that T2 = floor(c2/gamma) is the maximal depth used
DEPTH_FACTOR_2 = 1.0

# Max number of samples in the intermediate regime used for calculating the threshold error
MAX_NSHOT = 100

### Optimal parameters choice


def opt_params(
    target_error: float, noise_rate: float = 0.0, grid_search_width: int = 5
) -> tuple[int, int]:
    """
    Optimal parameters for the MS-QPE (Mulit-circuit Sine-state Quantum Phase Estimation)
    to achieve a given precision with global depolarising noise.

    Args:
        target_error (float): Desired precision for the phase estimation, quantified as the expected
            Holevo error [Eq. (A9)].
        noise_rate (float): Noise rate per application of the unitary [gamma in Eq. (B1)].
        grid_search_width (int): Width of grid search for optimizing circuit parameters.

    Returns:
        tuple:
            depth (int): Optimal depth for the circuit in terms of uses of the unitary [mathcal{T}
                in Alg. 14].
            n_samples (int): Number of measurement samples required to achieve the target error [M
                in Alg. 14].
    """

    thresh_depth1 = int(DEPTH_FACTOR_1 * noise_rate ** (-1 / 3))
    thresh_depth2 = int(DEPTH_FACTOR_2 / noise_rate)

    thresh_error1 = Holevo_error(thresh_depth1, noise_rate)
    thresh_error2 = 1 / np.sqrt(Fisher_information(thresh_depth2, noise_rate) * MAX_NSHOT)

    if target_error > thresh_error1:
        n_samples = 1
        depth = np.ceil(np.pi / np.arctan(target_error) - 2).astype(int)
    elif target_error < thresh_error2:
        n_samples = np.ceil(
            1 / Fisher_information(thresh_depth2, noise_rate) / target_error**2
        ).astype(int)
        depth = thresh_depth2
    else:
        # error_ratio = (target_error - thresh_error2) / (thresh_error1 - thresh_error2)
        # initial_guess_dim = thresh_depth1 + (thresh_depth2 - thresh_depth1) * (1 - error_ratio)
        # initial_guess = (initial_guess_dim, 10)
        depth, n_samples = _optimise_circuit_division(
            target_error,
            noise_rate,
            thresh_depth1 // 3,
            thresh_depth2,
            # initial_guess=initial_guess,
            grid_search_width=grid_search_width,
        )

    return depth, n_samples


### Ancillary functions


def _optimise_circuit_division(
    target_error: float,
    noise_rate: float,
    min_depth: int,
    max_depth: int,
    initial_guess: tuple[int, int] | None = None,
    grid_search_width: int = 5,
) -> tuple[int, int]:
    """
    Optimal parameters for MSQPE in the intermidiete regime.
    """
    if initial_guess is None:
        initial_guess = [min_depth + 1, 10]

    def constraint(x: tuple[int, int | float]) -> float:
        depth, n_sample = x
        return _error_bound(depth, n_sample, noise_rate) - target_error

    depth, n_samples = _discrete_minimize(
        fun=_cost,
        constraint=constraint,
        initial_guess=initial_guess,
        bounds=[(min_depth, max_depth), (1, np.infty)],
        grid_search_width=grid_search_width,
    )

    return depth, n_samples


##### Error model


def _error_bound(depth: int, n_samples: int | float, noise_rate: float) -> float:
    """
    Error bound for fixed circuit parameters [Eq. (B11)].
    """
    # Probability of failure
    a = -np.log(1 - np.exp(-noise_rate * depth)) / 2
    fail_prob = np.exp(-a * n_samples)
    # Error if failure
    fail_error = np.sqrt(2)

    if n_samples == 0:
        return fail_error

    # Error if success
    success_error = np.sqrt(1 / Fisher_information(int(depth), noise_rate) / n_samples)

    return np.sqrt(fail_prob * fail_error**2 + (1 - fail_prob) * success_error**2)


##### Cost function


def _cost(x: tuple[int | float, int | float]) -> int | float:
    depth, n_sample = x
    return depth * n_sample


##### Discrete optimization


def _discrete_minimize(
    fun: Callable[[tuple[float, float]], float],
    constraint: Callable[[tuple[int, int | float]], float],
    initial_guess: tuple[int, int],
    bounds: tuple[tuple[int | float, int | float], tuple[int | float, int | float]],
    grid_search_width: int = 5,
) -> tuple[int, int]:
    """
    Constrained minimization of a function of integers,
    by searching the grid around the minimum of the continuous version of the problem.
    """

    continuous_constraint = _interpolate_constraint(constraint)

    res = minimize(
        fun=fun,
        x0=initial_guess,
        bounds=bounds,
        constraints={"type": "eq", "fun": continuous_constraint},
    )

    if not res["success"]:
        print("Warning: optimization unsuccessful.")
        print(res)

    x_float = res["x"]

    x = _grid_search(_cost, x_float, grid_search_width, constraint)

    return x


def _interpolate_constraint(
    constraint: Callable[[int, int | float], float],
) -> Callable[[float, float], float]:
    """
    Transforms a function of a real number and an integer into a function of 2 reals numbers by interpolation.
    """

    def continuous_constraint(x):
        depth, n_samples = x
        int_depth = int(depth)
        interpolator = depth - int_depth
        err_0 = constraint((int_depth, n_samples))
        err_1 = constraint((int_depth + 1, n_samples))
        err = err_0 * (1 - interpolator) + err_1 * interpolator
        return err

    return continuous_constraint


def _grid_search(
    fun: Callable[[int, int | float], float],
    x0: tuple[float, float],
    width: int,
    constraint: Callable[[int, int | float], float],
) -> tuple[int, int]:
    """
    Brute-force constrained minimization by grid search around x0.
    """
    x0_int = np.round(x0)
    grid = (
        np.array(
            [[i, j] for i in np.arange(-width, width + 1) for j in np.arange(-width, width + 1)]
        )
        + x0_int
    )
    filtered_grid = grid[[constraint(x) < 0 and np.all(x > 0) for x in grid]]
    if len(filtered_grid) == 0:
        print(
            f"Constraint not satisfied anywhere around x0 = {x0},"
            f" minimal error: {np.min([constraint(x)<0 for x in grid])}"
        )
        return x0_int
    vals = [fun(x) for x in filtered_grid]
    x = filtered_grid[np.argmin(vals)]
    return x
