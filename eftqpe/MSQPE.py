import numpy as np
from scipy.optimize import minimize

from SinQPE import SinQPE_Holevo_error as Holevo_error
from SinQPE import SinQPE_FI as Fisher_information

### Algorithm constants for threshold errors calculation

# constant c1 such that T1 = floor(c1/gamma^(1/3)) is the maximal depth for which 1 sample is used
DEPTH_FACTOR_1 = (2 / 3 * np.pi**2) ** (1 / 3)

# constant c2 such that T2 = floor(c2/gamma) is the maximal depth used
DEPTH_FACTOR_2 = 1.0

# Max number of samples in the intermediate regime used for calculating the threshold error
MAX_NSHOT = 100

### Optimal parameters choice

def OptMSQPE_params(target_error, noise_rate=0.0, grid_search_width=5):
    thresh_depth1 = int(DEPTH_FACTOR_1 * noise_rate ** (-1 / 3))
    thresh_depth2 = int(DEPTH_FACTOR_2 / noise_rate)

    thresh_error1 = Holevo_error(thresh_depth1, noise_rate)
    thresh_error2 = 1 / np.sqrt(Fisher_information(thresh_depth2, noise_rate) * MAX_NSHOT)

    if target_error > thresh_error1:
        n_samples = 1
        depth = np.ceil(np.pi / target_error).astype(int)
    elif target_error < thresh_error2:
        n_samples = np.ceil(1 / Fisher_information(thresh_depth2, noise_rate) / target_error**2).astype(int)
        depth = thresh_depth2
    else:
        depth, n_samples = _optimise_circuit_division(
            target_error,
            noise_rate,
            thresh_depth1 // 3,
            thresh_depth2,
            grid_search_width = grid_search_width,
        )

    return int(depth), int(n_samples)

def _optimise_circuit_division(
    target_error, noise_rate, min_depth, max_depth, initial_guess=None, grid_search_width=5
):
    if initial_guess is None:
        initial_guess = [(min_depth + max_depth) / 2, 10]

    def constraint(x):
        depth, n_sample = x
        return _continuous_error(depth, n_sample, noise_rate) - target_error

    res = minimize(
        _cost,
        initial_guess,
        bounds=[(min_depth, max_depth), (1, np.infty)],
        constraints={"type": "eq", "fun": constraint},
    )

    if not res["success"]:
        print(
            f"Warning: optimization unsuccessful for target error {target_error}, noise rate {noise_rate}."
        )
        print(res)

    depth_float, n_samples_float = res["x"]

    depth, n_samples = _grid_minimization(
        _cost, (depth_float, n_samples_float), grid_search_width, constraint
    )

    return depth, n_samples

### Error model

def error_model(noise_rate, depth, n_samples):
    #Probability of failure
    a = -np.log(1 - np.exp(-noise_rate * depth)) / 2
    fail_prob = np.exp(-a * n_samples)
    #Error if failure
    fail_error = np.sqrt(2)
    #Error if success
    success_error = np.sqrt(1 / Fisher_information(int(depth), noise_rate) / n_samples)

    if n_samples == 0:
        return fail_error
    return np.sqrt(fail_prob * fail_error**2 + (1 - fail_prob) * success_error**2)


### Cost function

def _cost(x):
    depth, n_sample = x
    return depth * n_sample

### Discrete optimization

def _continuous_error(depth: float, n_sample: float, noise_rate):
    int_depth = int(depth)
    interpolator = depth - int_depth
    err_0 = error_model(noise_rate, int_depth, n_sample)
    err_1 = error_model(noise_rate, int_depth + 1, n_sample)
    err = err_0 * (1 - interpolator) + err_1 * interpolator
    return err

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




