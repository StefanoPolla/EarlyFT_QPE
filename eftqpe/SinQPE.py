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
    success_prob = np.exp(-depth * noise_rate)
    return (
        success_prob * noiseless_SinQPE_prob_func(x - true_phase, depth+2)
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


def SinQPE_FI(depth, noise_rate, integral_range_multiplier=10):
    if ~np.isfinite(depth) or depth < 1:
        return np.nan

    if noise_rate == 0.0:

        def integrand(x):
            return _dev_SinQPE_fun(x, depth + 2) ** 2 * SinQPE_prob_func(
                x, 0.0, depth, 0.0
            )

        FI = np.exp(-noise_rate * depth) * quad(integrand, -np.pi, np.pi)[0]
    else:

        def integrand(x):
            return dev_noiseless_SinQPE_prob_func(x, depth + 2) ** 2 / SinQPE_prob_func(
                x, 0.0, depth, noise_rate
            )

        int_bound = min(integral_range_multiplier / (depth+1), np.pi)

        FI = (
            np.exp(-2 * noise_rate * depth)
            * quad(integrand, -int_bound, int_bound, limit=100)[0]
        )

    return FI


def SinQPE_Holevo_error(depth, noise_rate=0.0):
    control_dim = depth + 1
    ests = np.arange(control_dim) * 2 * np.pi / control_dim
    probs = SinQPE_prob_vec(0.0, depth, noise_rate)
    return np.sqrt(np.sum(probs * (np.abs(np.exp(1j * ests) - 1)) ** 2))




