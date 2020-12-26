import numpy as np
from tinygrad.tensor import Tensor
class Transformer:

  class Encoder:
    pass 

  class EncoderLayer:
    pass

  class LayerNorm:
    def __init__(self, feats: int, eps: float = 1e-6):
      self.feats = feats
      self.eps = eps
      pass
    def forward(self, x):
      #n = x.shape[-1]
      n = self.feats
      w = Tensor(np.ones(n, dtype=np.float32))
      b = Tensor(np.zeros(n, dtype=np.float32))
      mu = x.mean(axis=-1)

      bessel_corr = n if 1 >= n else n-1
      std = x.sub(mu).pow(2).sum().div(Tensor(n).add(self.eps)).sqrt()
      # TODO: fix sum axis -1 w- reshape, for multi dim or channel tensors
      print("x - mu = {}".format(x.sub(mu)))
      print("(x - mu)/std = {}".format((x.sub(mu)).div(std)))
      print("mu: {}, std: {}".format(mu, std))

      return w.mul(x.sub(mu).div(std)).add(b)


  class SublayerConnection:
    def __init__(self, feats: int, dropout: float = 0.3):
      self.norm = Transformer.LayerNorm(feats)
      self.dropout = dropout
    def forward(self, x, sublayer):
      return x.add(sublayer.forward(self.norm.forward(x)).dropout(self.dropout))

  def __init__(self):
    pass
  def forward(self, x):
    feats = x.shape[-1]
    norm = self.LayerNorm(x.shape[-1])
    subc = self.SublayerConnection(feats, 0.)

    return subc.forward(x, norm)
  
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
