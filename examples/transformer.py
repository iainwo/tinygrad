import numpy as np
from tinygrad.tensor import Tensor
class Transformer:
  def __init__(self):
    pass
  def forward(self, x):
    # Layer Norm
    n = x.shape[-1]
    w = Tensor(np.ones(n, dtype=np.float32))
    b = Tensor(np.zeros(n, dtype=np.float32))
    mu = x.mean(axis=-1)

    
    bessel_corr = n if 1 >= n else n-1
    std = x.sub(mu).pow(2).sum().div(Tensor(n)).sqrt()
    print("mu: {}, std: {}".format(mu, std))

    return w.mul(x.sub(mu).div(std)).add(b)
  
if "__main__" == __name__:
  model = Transformer()
  x = Tensor(
    np.array([
      float(x) for x in range(10)
    ], dtype=np.float32)
  )
  y = model.forward(x)
  print("x: {}".format(x))
  print("y: {}".format(y))
