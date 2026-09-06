# Is Schulman's k3 KL estimator actually variance-optimal?

Is λ = 1 special?
Schulman's k3 estimator (http://joschu.net/blog/kl-approx.html) uses 
a fixed control-variate coefficient of 1 to obtain an unbiased, nonnegative, low-variance Monte Carlo
estimate of KL divergence. This project investigates how close this choice is the variance-optimal 
coefficient across across increasing policy shifts and wheter the coefficient can be estimated
online. 
