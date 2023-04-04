import numpy as np

def kl_divergence(p, q):
    # normalize the probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # ensure q[i] > 0 for all i where p[i] > 0
    mask = (p > 0) & (q == 0)
    q[mask] = np.finfo(float).eps

    # calculate the KL divergence
    kl = np.sum(p * np.log2(p / q))

    return kl
