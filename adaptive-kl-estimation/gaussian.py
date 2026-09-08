import torch 

# for reproducibility
torch.manual_seed(1023)
torch.set_default_dtype(torch.float64)


def run_experiment(mu: float, n_samples: int):
    P = torch.distributions.Normal(0.0, 1.0)
    Q = torch.distributions.Normal(mu, 1.0)


    # for monte-carlo estimation
    x = P.sample((n_samples, ))

    log_p = P.log_prob(x)
    log_q = Q.log_prob(x)

    # log(r): log(Q/P)
    log_r = log_q - log_p
    r = torch.exp(log_r) 

    b = r - 1.0

    cov_a_b = ((log_r - log_r.mean()) * (b - b.mean())).mean()
    var_b = ((b - b.mean())**2).mean()

    lambda_star = cov_a_b/var_b

    k1 = -log_r
    k2 = 1/2*(log_r ** 2)
    k3 = (r - 1.0) - log_r
    k_opt = lambda_star*b - log_r



    true_kl = 0.5 * mu**2

    print(f"true KL: {true_kl:.6f}")
    k4, lambda_loo = k4_leave_one_out(log_r, r)
    for name, values in [
        ("k1", k1),
        ("k2", k2),
        ("k3", k3),
        ("k-opt", k_opt),
        ("k4", k4)
    ]:
        print(
            f"{name}: "
            f"bias/true={(values.mean() - true_kl)/true_kl}, "
            f"stdev/true={values.std()/true_kl}, "

        )


def k4_leave_one_out(log_r, r, eps=1e-12):
    """
    Leave-one-out optimal control-variate KL estimator.

    For each sample i:
        k4_i = -log(r_i) + lambda_{-i} * (r_i - 1)

    where lambda_{-i} is estimated using every sample except i.
    """

    a = -log_r
    c = r - 1.0

    n = a.numel()

    if n < 3:
        raise ValueError("k4 requires at least 3 samples")

    # Sufficient statistics over full batch
    sum_a = a.sum()
    sum_c = c.sum()
    sum_ac = (a * c).sum()
    sum_c2 = (c ** 2).sum()

    # Number of observations in each leave-one-out set
    m = n - 1

    # Remove observation i
    sum_a_loo = sum_a - a
    sum_c_loo = sum_c - c

    # Unnormalized covariance numerator:
    #
    # sum_j!=i (a_j - mean_a)(c_j - mean_c)
    cov_num_loo = (
        sum_ac
        - a * c
        - (sum_a_loo * sum_c_loo) / m
    )

    # Unnormalized variance numerator:
    #
    # sum_j!=i (c_j - mean_c)^2
    var_num_loo = (
        sum_c2
        - c ** 2
        - (sum_c_loo ** 2) / m
    )

    lambda_loo = torch.where(
        var_num_loo > eps,
        -cov_num_loo / var_num_loo,
        torch.zeros_like(var_num_loo),
    )

    k4_terms = a + lambda_loo * c

    return k4_terms, lambda_loo




if __name__ == "__main__":
    run_experiment(mu=0.1, n_samples=1_000_000)
    run_experiment(mu=1, n_samples=1_000_000)

