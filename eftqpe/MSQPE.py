import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize, basinhopping
from eftqpe.utils import circ_dist

from SinQPE import SinQPE_prob_func as pmf
from SinQPE import SinQPE_Holevo_error as Holevo_error
from SinQPE import SinQPE_FI as Fisher_information

# constant c1 such that T1 = floor(c1/gamma^(1/3)) is the maximal depth for which 1 sample is used
DEPTH_FACTOR_1 = (2 / 3 * np.pi**2) ** (1 / 3)

# constant c2 such that T2 = floor(c2/gamma) is the maximal depth used
DEPTH_FACTOR_2 = 1.0

# Max number of samples in the intermediate regime used for calculating the threshold error
MAX_NSHOT = 100

def OptMLESinQPE_params(target_error, damp_strength=0.0, grid_search_width=5):
    thresh_dim1 = int(DEPTH_FACTOR_1 * damp_strength ** (-1 / 3)) + 1
    thresh_dim2 = int(DEPTH_FACTOR_2 / damp_strength) + 1

    thresh_error1 = Holevo_error(thresh_dim1, damp_strength)
    thresh_error2 = 1 / np.sqrt(Fisher_information(thresh_dim2, damp_strength) * MAX_NSHOT)

    if target_error > thresh_error1:
        n_samples = 1
        control_dim = 1 + np.ceil(np.pi / target_error).astype(int)
    elif target_error < thresh_error2:
        n_samples = np.ceil(1 / Fisher_information(thresh_dim2, damp_strength) / target_error**2).astype(int)
        control_dim = thresh_dim2
    else:
        control_dim, n_samples = _OptMLESinQPE_midregime(
            target_error,
            damp_strength,
            thresh_dim1,
            thresh_dim2,
            grid_search_width=grid_search_width,
        )

    return int(control_dim), int(n_samples)

def _OptMLESinQPE_midregime(
    target_error, damp_strength, thresh_dim1, thresh_dim2, initial_guess=None, grid_search_width=5
):
    if initial_guess is None:
        initial_guess = [(thresh_dim1 + thresh_dim2) / 2, 10]

    def constraint(x):
        control_dim, n_sample = x
        return _continuous_error(control_dim, n_sample, damp_strength) - target_error

    res = minimize(
        _cost,
        initial_guess,
        bounds=[(thresh_dim1 // 3, thresh_dim2), (1, np.infty)],
        constraints={"type": "eq", "fun": constraint},
    )

    if not res["success"]:
        print(
            f"Warning: optimization unsuccessful for target error {target_error}, damping strength {damp_strength}."
        )
        print(res)

    control_dim_float, n_samples_float = res["x"]

    control_dim, n_samples = _grid_minimization(
        _cost, (control_dim_float, n_samples_float), grid_search_width, constraint
    )

    return control_dim, n_samples

# Model specific functions

def loglikelihood(samples, control_dim, damp_strength):
    return lambda x: np.mean(
        np.log(pmf(np.array(samples), x, control_dim, damp_strength))
    )

def negloglikelihood(samples, control_dim, damp_strength):
    return lambda x: -loglikelihood(samples, control_dim, damp_strength)(x)

def MLESinQPE_var_model(damp_strength, control_dim, n_samples):
    a = -np.log(1 - np.exp(-damp_strength * (control_dim - 1))) / 2
    fail_prob = np.exp(-a * n_samples)
    FI = Fisher_information(int(control_dim), damp_strength)
    if n_samples == 0:
        return fail_prob * 2
    return fail_prob * 2 + (1 - fail_prob) * 1 / FI / n_samples

# Optimization

def bruteforce_minimize(f, N):
    ests = []
    vals = []
    for j in range(N):
        phase = j * 2 * np.pi / N
        res = minimize_scalar(
            f, method="bounded", bounds=(phase - 2 * np.pi / N, phase + 2 * np.pi / N)
        )
        ests.append(res["x"])
        vals.append(res["fun"])
    est = ests[np.argmin(vals)]
    return est

def _continuous_error(control_dim: float, n_sample: float, damp_strength):
    int_ctrldim = int(control_dim)
    interpolator = control_dim - int_ctrldim
    err_0 = np.sqrt(MLESinQPE_var_model(damp_strength, int_ctrldim, n_sample))
    err_1 = np.sqrt(MLESinQPE_var_model(damp_strength, int_ctrldim + 1, n_sample))
    err = err_0 * (1 - interpolator) + err_1 * interpolator
    return err


def _cost(x):
    control_dim, n_sample = x
    return (control_dim - 1) * n_sample


def _grid_minimization(fun, x0, width, constraint):
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
            f"Constraint not satisfied anywhere around x0 = {x0}, minimal error: {np.min([constraint(x)<0 for x in grid])}"
        )
        return x0_int
    vals = [fun(x) for x in filtered_grid]
    x = filtered_grid[np.argmin(vals)]
    return x




