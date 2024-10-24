def simulate_MLESinQPE(true_phase, control_dim, n_samples, damp_strength):
    samples = [
        sample_SinQPE(true_phase, control_dim, damp_strength, use_ref_phase)
        for i in range(n_samples)
    ]
    if n_samples > 1:
        est = bruteforce_minimize(f, control_dim)
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
    control_dim, n_samples = OptMLESinQPE_params(target_error, damp_strength, grid_search_width)
    return simulate_MLESinQPE(
        true_phase, control_dim, n_samples, damp_strength
    )

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




def loglikelihood(samples, control_dim, damp_strength):
    return lambda x: np.mean(
        np.log(SinQPE_prob_func(np.array(samples), x, control_dim, damp_strength))
    )


def negloglikelihood(samples, control_dim, damp_strength):
    return lambda x: -loglikelihood(samples, control_dim, damp_strength)(x)
