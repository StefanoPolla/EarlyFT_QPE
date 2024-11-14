import numpy as np
from scipy.integrate import quad


def SinQPE_Holevo_error(depth: int, noise_rate: float = 0.0) -> float:
    """
    Holevo error of the probability distribution of the Sin-state QPE circuit [Fig. 7]
    with global depolarising noise.

    Args:
        depth (int): Depth of the circuit in terms of uses of the unitary.
        noise_rate (float): Noise rate per application of the unitary [gamma in Eq. (B1)].

    Returns:
        float: Holevo error [(Eq. (A9))].
    """
    control_dim = depth + 1
    ests = np.arange(control_dim) * 2 * np.pi / control_dim
    probs = SinQPE_prob_vec(0.0, depth, noise_rate)
    return np.sqrt(np.sum(probs * (np.abs(np.exp(1j * ests) - 1)) ** 2))


def SinQPE_FI(depth: int, noise_rate: float, integral_range_multiplier: float = 10) -> float:
    """
    Fisher Information of the probability distribution of the Sin-state QPE circuit [Fig. 7]
    with global depolarising noise.

    Args:
        depth (int): Depth of the circuit in terms of uses of the unitary.
        noise_rate (float): Noise rate per application of the unitary [gamma in Eq. (B1)].
        integral_range_multiplier (float): Multiplier for the integration range for numerical
            stability.

    Returns:
        float: average Fisher Information [(Eq. (A16))].
    """
    if ~np.isfinite(depth) or depth < 1:
        return np.nan

    int_bound = min(integral_range_multiplier / (depth + 1), np.pi)

    if noise_rate == 0.0:

        def integrand(x):
            return derivative_log_SinQPE_func(x, depth) ** 2 * SinQPE_prob_func(x, depth, 0.0)

    else:

        def integrand(x):
            return derivative_noiseless_SinQPE_prob_func(x, depth) ** 2 / SinQPE_prob_func(
                x, depth, noise_rate
            )

    FI = np.exp(-2 * noise_rate * depth) * quad(integrand, -int_bound, int_bound, limit=100)[0]

    return FI


def SinQPE_prob_vec(true_phase: float, depth: int, noise_rate: float = 0.0) -> np.ndarray:
    """
    Probability vector for each bitsting of the Sin-state QPE circuit [Fig. 7]
    with global depolarising noise.

    Args:
        true_phase (float): True phase value to estimate.
        depth (int): Depth of the circuit in terms of uses of the unitary.
        noise_rate (float): Noise rate per application of the unitary [gamma in Eq. (B1)].

    Returns:
        np.ndarray: Probability vector of measurement outcomes.
    """
    control_dim = depth + 1
    x = np.arange(control_dim) * 2 * np.pi / control_dim

    return 2 * np.pi / control_dim * SinQPE_prob_func(x - true_phase, depth, noise_rate)


def SinQPE_prob_func(
    delta_phi: float | np.ndarray, depth: int, noise_rate: float = 0.0
) -> float | np.ndarray:
    """
    Average robability of a phase error delta_phi in the Sin-state QPE circuit [Fig. 7]
    with global depolarising noise.

    Args:
        delta_phi (float | np.ndarray): Phase difference (if the obtained bitstring is x, and the
            true value is phi, then delta_phi = 2pi x/(depth+1) - phi).
        depth (int): Depth of the circuit in terms of uses of the unitary.
        noise_rate (float): Noise rate per application of the unitary [gamma in Eq. (B1)].

    Returns:
        float | np.ndarray: Probability of observing the outcome(s) `x`.
    """
    success_prob = np.exp(-depth * noise_rate)
    return success_prob * noiseless_SinQPE_prob_func(delta_phi, depth) + (1 - success_prob) / (
        2 * np.pi
    )


def noiseless_SinQPE_prob_func(delta_phi: float | np.ndarray, depth: int) -> np.ndarray:
    """
    Continuous case of probability in the Sin-state QPE circuit [Fig. 7] without noise.

    Args:
        delta_phi (float | np.ndarray): Phase difference (if the obtained bitstring is x, and the
            true value is phi, then delta_phi = 2pi x/(depth+1) - phi).
        depth (int): Depth of the circuit in terms of uses of the unitary.

    Returns:
        float | np.ndarray: Probability of observing the outcome(s) `x`.
    """
    m = depth + 2
    v = (
        1
        / (2 * np.pi)
        * np.sin(np.pi / m) ** 2
        / m
        * (1 + np.cos(m * delta_phi))
        / (2 * np.sin(delta_phi / 2 + np.pi / (2 * m)) * np.sin(delta_phi / 2 - np.pi / (2 * m)))
        ** 2
    )
    v = np.array(v)
    v[np.cos(delta_phi) == np.cos(np.pi / m)] = m / (4 * np.pi)
    return v


def derivative_noiseless_SinQPE_prob_func(delta_phi: float | np.ndarray, depth: int) -> np.ndarray:
    """
    Derivative of the noiseless probability function for the Sin-state QPE circuit [Fig. 7]
    with respect to the true phase.

    Args:
        delta_phi (float | np.ndarray): Phase difference (if the obtained bitstring is x, and the
            true value is phi, then delta_phi = 2pi x/(depth+1) - phi).
        depth (int): Depth of the circuit in terms of uses of the unitary.

    Returns:
        float | np.ndarray: Derivative of the probability function for `x`.
    """
    return derivative_log_SinQPE_func(delta_phi, depth) * noiseless_SinQPE_prob_func(
        delta_phi, depth
    )


def derivative_log_SinQPE_func(delta_phi: float | np.ndarray, depth: int) -> np.ndarray:
    """
    Derivative of logarithm of the noiseless probability function for the Sin-state QPE circuit
    [Fig. 7] with respect to the true phase.

    Args:
        delta_phi (float | np.ndarray): Phase difference (if the obtained bitstring is x, and the
            true value is phi, then delta_phi = 2pi x/(depth+1) - phi).
        depth (int): Depth of the circuit in terms of uses of the unitary.

    Returns:
        float | np.ndarray: Derivative of the logarithm of the probability function for `x`.
    """
    m = depth + 2
    v = -m * np.tan(m * delta_phi / 2) - np.sin(delta_phi) / (
        np.sin(delta_phi / 2 + np.pi / (2 * m)) * np.sin(delta_phi / 2 - np.pi / (2 * m))
    )
    v = np.array(v)
    v[np.cos(delta_phi) == np.cos(np.pi / m)] = -1 / np.tan(np.pi / m)
    return v
