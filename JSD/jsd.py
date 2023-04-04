import numpy as np

def jensen_shannon_divergence(p, q):
    # normalize the probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # calculate the average distribution
    m = (p + q) / 2

    # calculate the Kullback-Leibler divergence between p and m
    kl_pm = np.sum(p * np.log2(p / m))

    # calculate the Kullback-Leibler divergence between q and m
    kl_qm = np.sum(q * np.log2(q / m))

    # calculate the Jensen-Shannon divergence
    jsd = (kl_pm + kl_qm) / 2

    return jsd
