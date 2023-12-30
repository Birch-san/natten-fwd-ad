from torch import FloatTensor
from torch.nested import nested_tensor
from torch.nn import Module, Linear
from einops import rearrange
from typing import NamedTuple, List

class QKV(NamedTuple):
  q: FloatTensor
  k: FloatTensor
  v: FloatTensor

from natten.functional import natten2dqk, natten2dav
# from src.natten_autograd import natten2dqk, natten2dav

# Simplifed version of
# Katherine Crowson's NeighborhoodSelfAttentionBlock, MIT license
# https://github.com/crowsonkb/k-diffusion/blob/9737cfd85120cba1258b5b5b1dc6511356b5c924/k_diffusion/models/image_transformer_v2.py#L390
class NattenBlock(Module):
  def __init__(self, d_model: int, d_head: int, kernel_size: int):
    super().__init__()
    self.d_head = d_head
    self.n_heads = d_model // d_head
    self.kernel_size = kernel_size
    self.qkv_proj = Linear(d_model, d_model * 3, bias=False)
    self.out_proj = Linear(d_model, d_model, bias=False)
  
  def _x_to_qkv(self, x: FloatTensor) -> QKV:
    qkv = self.qkv_proj(x)
    q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
    q = q / self.d_head**.5
    return QKV(q, k, v)
  
  def _av_to_out(self, x: FloatTensor) -> FloatTensor:
    x = rearrange(x, "n nh h w e -> n h w (nh e)")
    x = self.out_proj(x)
    return x

  def forward(self, x: FloatTensor) -> FloatTensor:
    if x.is_nested:
      qkvs: List[QKV] = [self._x_to_qkv(t) for t in x.unbind()]
      q = nested_tensor([qkv.q for qkv in qkvs])
      k = nested_tensor([qkv.k for qkv in qkvs])
      v = nested_tensor([qkv.v for qkv in qkvs])
    else:
      q, k, v = self._x_to_qkv(x)
    qk = natten2dqk(q, k, self.kernel_size, 1)
    a = qk.softmax(dim=-1)
    x = natten2dav(a, v, self.kernel_size, 1)
    if a.is_nested:
      x = nested_tensor([self._av_to_out(x_) for x_ in x.unbind()])
    else:
      x = self._av_to_out(x)
    return x