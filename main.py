import numpy as np

# import the KL divergence and Jensen-Shannon distance functions
from kl_divergence import kl_divergence
from jensen_shannon_divergence import jensen_shannon_divergence

# define the probability distributions p and q
p = np.array([0.3, 0.4, 0.3])
q = np.array([0.2, 0.5, 0.3])

# calculate the KL divergence between p and q
kl = kl_divergence(p, q)
print(f"The KL divergence between p and q is: {kl:.4f}")

# calculate the Jensen-Shannon distance between p and q
jsd = jensen_shannon_divergence(p, q)
print(f"The Jensen-Shannon distance between p and q is: {jsd:.4f}")
