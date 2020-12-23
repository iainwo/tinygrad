import numpy as np
from tinygrad.tensor import Tensor
class Transformer:
  def __init__(self):
    pass
  def forward(self, x):
    # Layer Norm
    n = 10
    w = Tensor(np.ones(n))
    b = Tensor(np.zeros(n))
    mu = x.mean(axis=-1)
    std = (x - mu).pow(2).div(Tensor(n))

    return w.mul(x.sub(mu).div(std)).add(b)
  
if "__main__" == __name__:
  model = Transformer()
  y = model.forward(Tensor(np.array([float(x) for x in range(10)], dtype=np.float32)))
  print(y)

