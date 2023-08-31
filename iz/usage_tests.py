# import torch
# from iz.metalog import metalog
# import numpy as np

# x = (torch.tensor([40, 50, 60, 70], dtype=torch.float32)).reshape(4, 1)
# probs = (torch.tensor([0.1, 0.5, 0.7, 0.9], dtype=torch.float32)).reshape(4, 1)

# my_metalog = metalog(
#     y=x,
#     probs=probs,
#     boundedness="u",
#     terms=4,
#     penalty=None,
# )

import pandas as pd
import iz
# import matplotlib.pyplot as plt

df = pd.read_csv("iz/data/fishSize.csv")
fish_metalog = iz.metalog(y=df.FishSize, bounds=[0,60], boundedness='b', terms=3, step_len=.001, epochs=500, lr = 0.1)
iz.summary(fish_metalog)
print(df.sort_values(by='FishSize').reset_index(drop=True))

# plt.hist(df.FishSize, 12)

r_gens = iz.rmetalog(fish_metalog, n=5000, generator="hdr")
# plt.hist(r_gens, 12)
# plt.show()


# quantiles from a percentile
qs = iz.qmetalog(fish_metalog, y=[0.25, 0.5, 0.999])
print("qmetalog demo: " + str(qs))

# probabilities from a quantile
ps = iz.pmetalog(fish_metalog, q=[3, 10, 35])
print("pmetalog demo: " + str(ps))

# density from a quantile
ds = iz.dmetalog(fish_metalog, q=[3, 10, 25])
print("dmetalog demo: " + str(ds))