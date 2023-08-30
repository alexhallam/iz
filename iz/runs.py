import torch
from iz.metalog import metalog
import numpy as np

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

df = pd.read_csv("iz/data/fishSize.csv")
fish_metalog = metalog(y=df.FishSize, bounds=[0], boundedness='sl', terms=4, step_len=.01, epochs=10000, lr = 0.1)
