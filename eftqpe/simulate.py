import numpy as np
from MSQPE import OptMSQPE_params
from scipy.optimize import minimize_scalar
from SinQPE import SinQPE_prob_func, SinQPE_prob_vec

from eftqpe.utils import circ_dist


def simulate_MLESinQPE(true_phase, control_dim, n_samples, damp_strength):
    samples = [
        sample_SinQPE(true_phase, control_dim, damp_strength, use_ref_phase=False)
        for i in range(n_samples)
    ]
    if n_samples > 1:
        est = bruteforce_minimize(
            negloglikelihood(samples, control_dim, damp_strength), control_dim
        )
    else:
        est = samples[0]
    error = circ_dist(est, true_phase)

    cost = (control_dim - 1) * n_samples

    return (error, cost)


def simulate_OptMLESinQPE(
    true_phase,
    target_error,
    damp_strength=0.0,
    grid_search_width=5,
):
    control_dim, n_samples = OptMSQPE_params(target_error, damp_strength, grid_search_width)
    return simulate_MLESinQPE(true_phase, control_dim, n_samples, damp_strength)


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


def SinQPE_loglikelihood(samples, depth, noise_rate):
    return lambda x: np.mean(np.log(SinQPE_prob_func(np.array(samples), x, depth, noise_rate)))


def negloglikelihood(samples, control_dim, damp_strength):
    return lambda x: -SinQPE_loglikelihood(samples, control_dim, damp_strength)(x)


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
