import numpy as np
from scipy.integrate import quad

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


def SinQPE_prob_vec(phase, depth, noise_rate=0.0):
    # Discrete case
    control_dim = depth + 1
    success_prob = np.exp(-depth * noise_rate)

    diffs = phase - np.arange(control_dim) * 2 * np.pi / control_dim
    noiseless_probs = 2 * np.pi / control_dim * noiseless_SinQPE_prob_func(diffs, control_dim + 1)

    return success_prob * noiseless_probs + (1 - success_prob) * np.ones(control_dim) / control_dim


def SinQPE_prob_func(x, true_phase, depth, noise_rate=0.0):
    # Continuous case
    control_dim = depth + 1
    success_prob = np.exp(-depth * noise_rate)
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


def SinQPE_FI(control_dim, noise_rate, integral_range_multiplier=10):
    if ~np.isfinite(control_dim) or control_dim < 0:
        return np.nan

    if noise_rate == 0.0:

        def integrand(x):
            return _dev_SinQPE_fun(x, control_dim + 1) ** 2 * SinQPE_prob_func(
                x, 0.0, control_dim, 0.0
            )

        FI = np.exp(-noise_rate * (control_dim - 1)) * quad(integrand, -np.pi, np.pi)[0]
    else:

        def integrand(x):
            return dev_noiseless_SinQPE_prob_func(x, control_dim + 1) ** 2 / SinQPE_prob_func(
                x, 0.0, control_dim, noise_rate
            )

        int_bound = min(integral_range_multiplier / control_dim, np.pi)

        FI = (
            np.exp(-2 * noise_rate * (control_dim - 1))
            * quad(integrand, -int_bound, int_bound, limit=100)[0]
        )

    return FI


def SinQPE_Holevo_error(control_dim, noise_rate=0.0):
    ests = np.arange(control_dim) * 2 * np.pi / control_dim
    probs = SinQPE_prob_vec(0.0, control_dim, noise_rate)
    return np.sqrt(np.sum(probs * (np.abs(np.exp(1j * ests) - 1)) ** 2))




