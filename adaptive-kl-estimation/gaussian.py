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

    k1 = -log_r
    k2 = 1/2*(log_r ** 2)
    k3 = (r - 1.0) - log_r


    true_kl = 0.5 * mu**2

    print(f"true KL: {true_kl:.6f}")

    for name, values in [
        ("k1", k1),
        ("k2", k2),
        ("k3", k3),
    ]:
        print(
            f"{name}: "
            f"mean={values.mean().item():.6f}, "
            f"std={values.std().item():.6f}, "
            f"bias={values.mean().item() - true_kl:.6f}"

        )

if __name__ == "__main__":
    run_experiment(mu=0.1, n_samples=1_000_000)

