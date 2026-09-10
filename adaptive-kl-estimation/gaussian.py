from pathlib import Path

import matplotlib.pyplot as plt
import torch

# for reproducibility
torch.manual_seed(1023)
torch.set_default_dtype(torch.float64)


def run_experiment(mu: float, batch_size: int, n_trials: int = 10_000):
    if mu == 0:
        raise ValueError("Normalized metrics require a nonzero true KL (mu != 0)")
    if batch_size < 3:
        raise ValueError("k4 requires a batch size of at least 3")
    if n_trials < 2:
        raise ValueError("Standard deviation requires at least 2 trials")

    P = torch.distributions.Normal(0.0, 1.0)
    Q = torch.distributions.Normal(mu, 1.0)


    # Each row is an independent mini-batch estimate.
    x = P.sample((n_trials, batch_size))

    log_p = P.log_prob(x)
    log_q = Q.log_prob(x)

    # log(r): log(Q/P)
    log_r = log_q - log_p
    r = torch.exp(log_r) 

    b = r - 1.0

    # calculate lambda_star for each minibatch

    mean_log_r = log_r.mean(dim=1, keepdim=True)
    mean_b = b.mean(dim=1, keepdim=True)

    cov_logr_b = ((log_r - mean_log_r) * (b - mean_b)).mean(dim=1, keepdim=True)
    var_b = ((b - mean_b)**2).mean(dim=1, keepdim=True)

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

    print(f"mu={mu}, batch size={batch_size}, trials={n_trials:,}, true KL={true_kl:.6f}")
    results = {}
    for name, values in [
        ("k1", k1_batch_estimate),
        ("k2", k2_batch_estimate),
        ("k3", k3_batch_estimate),
        ("k-opt", k_opt_batch_estimate),
        ("kloo", k_loo_batch_estimate)
    ]:
        bias_ratio = ((values.mean() - true_kl) / true_kl).item()
        stdev_ratio = (values.std() / true_kl).item()
        rmse_ratio = (
            torch.sqrt(((values - true_kl)**2).mean())/true_kl
        ).item()

        results[name] = {"bias/true": bias_ratio, "stdev/true": stdev_ratio, "rmse/true": rmse_ratio}
        print(
            f"{name}: "
            f"bias/true={bias_ratio}, "
            f"stdev/true={stdev_ratio}, "
            f"rmse/true={rmse_ratio}"
        )

    return results


def plot_results(experiments, n_trials, output_path):
    metrics = ("bias/true", "stdev/true", "rmse/true")
    fig, axes = plt.subplots(
        len(experiments), len(metrics), figsize=(16, 4 * len(experiments)),
        squeeze=False, layout="constrained",
    )
    colors = ["#3274a1", "#e1812c", "#3a923a", "#c03d3e", "#9372b2"]
    markers = ["o", "s", "^", "D", "x"]
    for row, (mu, batch_results) in enumerate(experiments.items()):
        batch_sizes = sorted(batch_results)
        estimators = list(batch_results[batch_sizes[0]])
        for column, metric in enumerate(metrics):
            ax = axes[row, column]
            for name, color, marker in zip(estimators, colors, markers):
                values = [batch_results[size][name][metric] for size in batch_sizes]
                ax.plot(batch_sizes, values, label=name, color=color, marker=marker)
            ax.set_xscale("log", base=2)
            ax.set_xticks(batch_sizes, labels=[str(size) for size in batch_sizes])
            if metric == "bias/true":
                ax.axhline(0, color="black", linewidth=0.8)
            else:
                ax.set_yscale("log")
            ax.set_title(f"mu = {mu}, true KL = {0.5 * mu**2:g}")
            ax.set_xlabel("Mini-batch size")
            ax.set_ylabel(metric)
            ax.set_axisbelow(True)
            ax.grid(alpha=0.25)
            ax.margins(y=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8)

    fig.suptitle(f"Gaussian KL estimators ({n_trials:,} independent trials per mini-batch size)")
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

    n = a.shape[1]

    if n < 3:
        raise ValueError("k4 requires at least 3 samples")

    # stats over batch
    sum_a = a.sum(dim=1, keepdim=True)
    sum_c = c.sum(dim=1, keepdim=True)
    sum_ac = (a * c).sum(dim=1, keepdim=True)
    sum_c2 = (c ** 2).sum(dim=1, keepdim=True)

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
    n_trials = 10_000
    batch_sizes = [2**power for power in range(2, 9)]
    experiments = {
        mu: {
            batch_size: run_experiment(mu=mu, batch_size=batch_size, n_trials=n_trials)
            for batch_size in batch_sizes
        }
        for mu in (0.1, 1)
    }
    plot_results(
        experiments, n_trials, Path(__file__).with_name("gaussian_batch_metrics.png")
    )
