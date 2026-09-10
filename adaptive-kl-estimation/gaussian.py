from pathlib import Path

import matplotlib.pyplot as plt
import torch

# for reproducibility
torch.manual_seed(1023)
torch.set_default_dtype(torch.float64)


def run_experiment(mu: float, n_samples: int, batch_size: int):
    P = torch.distributions.Normal(0.0, 1.0)
    Q = torch.distributions.Normal(mu, 1.0)


    # for batch estimation
    n_trials = 10_000


    x = P.sample((n_trials, batch_size)) # size: [10000, 64]

    log_p = P.log_prob(x)
    log_q = Q.log_prob(x)

    # log(r): log(Q/P)
    log_r = log_q - log_p
    r = torch.exp(log_r) 

    b = r - 1.0

    cov_logr_b = ((log_r - log_r.mean()) * (b - b.mean())).mean()
    var_b = ((b - b.mean())**2).mean()

    lambda_star = cov_logr_b/var_b

    k1 = -log_r
    k1_batch_estimate = k1.mean(dim=1)
    k2 = 1/2*(log_r ** 2)
    k2_batch_estimate = k2.mean(dim=1)
    k3 = (r - 1.0) - log_r
    k3_batch_estimate = k3.mean(dim=1)
    k_opt = lambda_star*b - log_r
    k_opt_batch_estimate = k_opt.mean(dim=1)
    k_loo, lambda_loo = k_leave_one_out(log_r=log_r, r=r)
    k_loo_batch_estimate = k_loo.mean(dim=1)



    true_kl = 0.5 * mu**2

    print(f"true KL: {true_kl:.6f}")
    results = {}
    for name, values in [
        ("k1", k1_batch_estimate),
        ("k2", k2_batch_estimate),
        ("k3", k3_batch_estimate),
        ("k-opt", k_opt_batch_estimate),
        ("k4", k4)
    ]:
        bias_ratio = ((values.mean() - true_kl) / true_kl).item()
        stdev_ratio = (values.std() / true_kl).item()
        results[name] = {"bias/true": bias_ratio, "stdev/true": stdev_ratio}
        print(
            f"{name}: "
            f"bias/true={bias_ratio}, "
            f"stdev/true={stdev_ratio}, "
        )

    return results


def plot_results(experiments, n_samples, output_path):
    fig, axes = plt.subplots(
        len(experiments), 2, figsize=(12, 4 * len(experiments)),
        squeeze=False, layout="constrained",
    )
    colors = ["#3274a1", "#e1812c", "#3a923a", "#c03d3e", "#9372b2"]
    for row, (mu, results) in enumerate(experiments.items()):
        for column, metric in enumerate(("bias/true", "stdev/true")):
            ax = axes[row, column]
            values = [metrics[metric] for metrics in results.values()]
            bars = ax.bar(list(results), values, color=colors)
            ax.bar_label(bars, labels=[f"{value:.4g}" for value in values], padding=4)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"mu = {mu}, true KL = {0.5 * mu**2:g}")
            ax.set_ylabel(metric)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.25)
            ax.margins(y=0.2)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Gaussian KL estimators ({n_samples:,} samples)")
    fig.savefig(output_path, dpi=200)
    print(f"Chart saved to {output_path}")
    plt.show()


def k_leave_one_out(log_r, r, eps=1e-12):
    """
    Leave-one-out optimal control-variate KL estimator.

    For each sample i:
        k_loo = -log(r_i) + lambda_{-i} * (r_i - 1)

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

    k_loo = a + lambda_loo * c
    


    return k_loo, lambda_loo




if __name__ == "__main__":
    n_samples = 1_000_000
    experiments = {
        mu: run_experiment(mu=mu, n_samples=n_samples)
        for mu in (0.1, 1)
    }
    plot_results(
        experiments, n_samples, Path(__file__).with_name("gaussian_metrics.png")
    )
