#poetry run pytest tests/test.py 

import torch
from lambda_gamma.metalog import metalog
import numpy as np

x = (torch.tensor([40, 50, 70], dtype=torch.float32)).reshape(3, 1)
probs = (torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)).reshape(3, 1)
# numpy version
#x = np.array([40, 50, 70])
#probs = np.array([0.1, 0.5, 0.9])

my_metalog = metalog(
    x=x,
    probs=probs,
    boundedness="sl",
    bounds=[0],
    terms=3,
    penalty=None,
)