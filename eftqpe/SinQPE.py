"""
TODO: document the functions that need to be used externally.

TODO: possibly split this into multiple files for better organization.
possible division:
- Fisher information
- Optimal parameters choice
- MLE simulation
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize, basinhopping
from eftqpe.utils import circ_dist

# constant c1 such that T1 = floor(c1/gamma^(1/3)) is the maximal depth for which 1 sample is used
DEPTH_FACTOR_1 = (2 / 3 * np.pi**2) ** (1 / 3)

# constant c2 such that T2 = floor(c2/gamma) is the maximal depth used
DEPTH_FACTOR_2 = 1.0

# Max number of samples in the intermediate regime used for calculating the threshold error
MAX_NSHOT = 100


def noiseless_SinQPE_prob_func(x, m):
    v = np.array(
        1
        / 2
        / np.pi
        * np.sin(np.pi / m) ** 2
        / m
        * (1 + np.cos(m * x))
        / (2 * np.sin(x / 2 + np.pi / 2 / m) * np.sin(x / 2 - np.pi / 2 / m)) ** 2
    )
    v[np.cos(x) == np.cos(np.pi / m)] = m / 4 / np.pi
    return v


def SinQPE_prob_vec(phase, control_dim, damp_strength=0.0):
    # Discrete case
    tmax = control_dim - 1
    success_prob = np.exp(-tmax * damp_strength)

    diffs = phase - np.arange(control_dim) * 2 * np.pi / control_dim
    noiseless_probs = 2 * np.pi / control_dim * noiseless_SinQPE_prob_func(diffs, control_dim + 1)

    return success_prob * noiseless_probs + (1 - success_prob) * np.ones(control_dim) / control_dim


def SinQPE_prob_func(x, true_phase, control_dim, damp_strength=0.0):
    # Continuous case
    depth = control_dim - 1
    success_prob = np.exp(-depth * damp_strength)
    return (
        success_prob * noiseless_SinQPE_prob_func(x - true_phase, control_dim + 1)
        + (1 - success_prob) / 2 / np.pi
    )


def _dev_SinQPE_fun(x, m):
    # f'(x) = f(x)*g(x)
    v = np.array(
        (
            -m * np.tan(m * x / 2)
            - np.sin(x) / np.sin(x / 2 + np.pi / 2 / m) / np.sin(x / 2 - np.pi / 2 / m)
        )
    )
    v[np.cos(x) == np.cos(np.pi / m)] = -1 / np.tan(np.pi / m)
    return v


def dev_noiseless_SinQPE_prob_func(x, m):
    return _dev_SinQPE_fun(x, m) * noiseless_SinQPE_prob_func(x, m)


def SinQPE_FI(control_dim, damp_strength, integral_range_multiplier=10):
    if ~np.isfinite(control_dim) or control_dim < 0:
        return np.nan

    if damp_strength == 0.0:

        def integrand(x):
            return _dev_SinQPE_fun(x, control_dim + 1) ** 2 * SinQPE_prob_func(
                x, 0.0, control_dim, 0.0
            )

        FI = np.exp(-damp_strength * (control_dim - 1)) * quad(integrand, -np.pi, np.pi)[0]
    else:

        def integrand(x):
            return dev_noiseless_SinQPE_prob_func(x, control_dim + 1) ** 2 / SinQPE_prob_func(
                x, 0.0, control_dim, damp_strength
            )

        int_bound = min(integral_range_multiplier / control_dim, np.pi)

        FI = (
            np.exp(-2 * damp_strength * (control_dim - 1))
            * quad(integrand, -int_bound, int_bound, limit=100)[0]
        )

    return FI


def SinQPE_Holevo_error(control_dim, damp_strength=0.0):
    ests = np.arange(control_dim) * 2 * np.pi / control_dim
    probs = SinQPE_prob_vec(0.0, control_dim, damp_strength)
    return np.sqrt(np.sum(probs * (np.abs(np.exp(1j * ests) - 1)) ** 2))


def sample_SinQPE(true_phase, control_dim, damp_strength=0.0, use_ref_phase=True):
    ref_phase = 0.0
    if use_ref_phase:
        ref_phase = np.random.uniform(0, 2 * np.pi)
    probs = SinQPE_prob_vec(true_phase - ref_phase, control_dim, damp_strength)

    if np.abs(1 - np.sum(probs)) > 1e-4:
        print(f"Warning: Probs sum to {np.sum(probs)}")
    probs = probs / np.sum(probs)

    ests = np.arange(control_dim) * 2 * np.pi / control_dim
    est = np.random.choice(ests, p=probs)
    return (est + ref_phase) % (2 * np.pi)


def simulate_meanSinQPE(true_phase, control_dim, n_samples, damp_strength=0.0, use_ref_phase=True):
    samples = [
        np.exp(1j * sample_SinQPE(true_phase, control_dim, damp_strength, use_ref_phase))
        for i in range(n_samples)
    ]
    est = np.angle(np.mean(samples))
    error = circ_dist(est, true_phase)

    cost = (control_dim - 1) * n_samples

    return (error, cost)


def simulate_medianSinQPE(
    true_phase, control_dim, n_samples, damp_strength=0.0, use_ref_phase=True
):
    samples = [
        sample_SinQPE(true_phase, control_dim, damp_strength, use_ref_phase)
        for i in range(n_samples)
    ]
    est = np.median(samples)
    error = circ_dist(est, true_phase)

    cost = (control_dim - 1) * n_samples

    return (error, cost)


def OptMeanSinQPE_params(target_error, damp_strength=0.0):
    thresh_error = (np.pi * damp_strength) ** (1 / 3)

    if target_error > thresh_error:
        n_samples = 1
        control_dim = np.ceil(np.pi / target_error).astype(int)
    else:
        n_samples = np.ceil((thresh_error / target_error) ** 2).astype(int)
        control_dim = np.floor((np.pi**2 / damp_strength) ** (1 / 3)).astype(int)
    return control_dim, n_samples


def simulate_OptMeanSinQPE(true_phase, target_error, damp_strength=0.0):
    control_dim, n_samples = OptMeanSinQPE_params(target_error, damp_strength)
    return simulate_meanSinQPE(true_phase, control_dim, n_samples, damp_strength)


def loglikelihood(samples, control_dim, damp_strength):
    return lambda x: np.mean(
        np.log(SinQPE_prob_func(np.array(samples), x, control_dim, damp_strength))
    )


def negloglikelihood(samples, control_dim, damp_strength):
    return lambda x: -loglikelihood(samples, control_dim, damp_strength)(x)


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


def local_minimize(f, N, true_phase):
    return minimize_scalar(
        f, method="bounded", bounds=(true_phase - 2 * np.pi / N, true_phase + 2 * np.pi / N)
    )["x"]


def simulate_MLESinQPE(true_phase, control_dim, n_samples, damp_strength, minimizer, use_ref_phase):
    samples = [
        sample_SinQPE(true_phase, control_dim, damp_strength, use_ref_phase)
        for i in range(n_samples)
    ]
    if n_samples > 1:
        f = negloglikelihood(samples, control_dim, damp_strength)
        if minimizer == "bruteforce":
            est = bruteforce_minimize(f, control_dim)
        elif minimizer == "local":
            est = local_minimize(f, control_dim, true_phase)
        elif minimizer == "simple":
            est = minimize_scalar(f, method="bounded", bounds=(0, 2 * np.pi))["x"]
        elif minimizer == "basinhopping":
            est = basinhopping(f, np.angle(np.mean(samples)))["x"][0]
    else:
        est = samples[0]
    error = circ_dist(est, true_phase)

    cost = (control_dim - 1) * n_samples

    return (error, cost)


def MLESinQPE_var_model(damp_strength, control_dim, n_samples):
    a = -np.log(1 - np.exp(-damp_strength * (control_dim - 1))) / 2
    fail_prob = np.exp(-a * n_samples)
    FI = SinQPE_FI(int(control_dim), damp_strength)
    if n_samples == 0:
        return fail_prob * 2
    return fail_prob * 2 + (1 - fail_prob) * 1 / FI / n_samples


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


def OptMLESinQPE_params(target_error, damp_strength=0.0, grid_search_width=5):
    thresh_dim1 = int(DEPTH_FACTOR_1 * damp_strength ** (-1 / 3)) + 1
    thresh_dim2 = int(DEPTH_FACTOR_2 / damp_strength) + 1

    thresh_error1 = SinQPE_Holevo_error(thresh_dim1, damp_strength)
    thresh_error2 = 1 / np.sqrt(SinQPE_FI(thresh_dim2, damp_strength) * MAX_NSHOT)

    if target_error > thresh_error1:
        n_samples = 1
        control_dim = 1 + np.ceil(np.pi / target_error).astype(int)
    elif target_error < thresh_error2:
        n_samples = np.ceil(1 / SinQPE_FI(thresh_dim2, damp_strength) / target_error**2).astype(int)
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


def simulate_OptMLESinQPE(
    true_phase,
    target_error,
    damp_strength=0.0,
    minimizer="bruteforce",
    use_ref_phase=True,
    grid_search_width=5,
):
    control_dim, n_samples = OptMLESinQPE_params(target_error, damp_strength, grid_search_width)
    return simulate_MLESinQPE(
        true_phase, control_dim, n_samples, damp_strength, minimizer, use_ref_phase
    )
