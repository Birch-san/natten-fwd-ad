from torch import FloatTensor
from torch.nn import Module, Linear
from torch.nn.functional import linear
from einops import rearrange
from typing import Optional

from natten.functional import na2d_qk, na2d_av

# Simplifed version of
# Katherine Crowson's NeighborhoodSelfAttentionBlock, MIT license
# https://github.com/crowsonkb/k-diffusion/blob/9737cfd85120cba1258b5b5b1dc6511356b5c924/k_diffusion/models/image_transformer_v2.py#L390
class NattenRegistersBlock(Module):
  def __init__(self, d_model: int, d_head: int, kernel_size: int):
    super().__init__()
    self.d_model = d_model
    self.d_head = d_head
    self.n_heads = d_model // d_head
    self.kernel_size = kernel_size
    self.qkv_proj = Linear(d_model, d_model * 3, bias=False)
    self.out_proj = Linear(d_model, d_model, bias=False)

  def forward(self, x: FloatTensor, registers: FloatTensor = None) -> FloatTensor:
    qkv = self.qkv_proj(x)

    # this has nothing to do with registers, I'm just slicing a kv_proj out of qkv_proj
    kv_proj_weight: FloatTensor = self.qkv_proj.weight[self.d_model:, :]
    kv_proj_bias: Optional[FloatTensor] = None if self.qkv_proj.bias is None else self.qkv_proj.bias[self.d_model:]

    # project the registers through kv_proj (a separate kv_proj would be fine too)
    reg_kv: FloatTensor = linear(registers, kv_proj_weight, kv_proj_bias)
    # permute head channels into a batch dim, and unbind k and v from the fused kv
    reg_k, reg_v = rearrange(reg_kv, "r (t nh e) -> t 1 nh r e", t=2, e=self.d_head)
    # expand singleton batch dim to explicitly equal batch dim used in self-attn
    reg_k, reg_v = (reg.expand(qkv.size(0), *reg.shape[1:]) for reg in (reg_k, reg_v))
    # repeat singleton batch dim over batch dim used in self-attn
    # reg_k, reg_v = (reg.repeat(qkv.size(0), *(1,)*(reg.ndim-1)) for reg in (reg_k, reg_v))

    q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
    q = q / self.d_head**.5
    qk = na2d_qk(q, k, self.kernel_size, 1, additional_keys=reg_k)
    a = qk.softmax(dim=-1)
    x = na2d_av(a, v, self.kernel_size, 1, additional_values=reg_v)
    x = rearrange(x, "n nh h w e -> n h w (nh e)")
    x = self.out_proj(x)
    return x