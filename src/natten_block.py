from torch import FloatTensor
from torch.nn import Module, Linear
from einops import rearrange

from natten.functional import natten2dqk, natten2dav
# from src.natten_autograd import natten2dqk, natten2dav

# Simplifed version of
# Katherine Crowson's NeighborhoodSelfAttentionBlock, MIT license
# https://github.com/crowsonkb/k-diffusion/blob/9737cfd85120cba1258b5b5b1dc6511356b5c924/k_diffusion/models/image_transformer_v2.py#L390
class NattenBlock(Module):
  def __init__(self, d_model: int, d_head: int, kv_groups: int, kernel_size: int):
    super().__init__()
    self.d_head = d_head
    self.q_heads = d_model // d_head
    self.kv_groups = kv_groups
    assert self.q_heads % kv_groups == 0
    self.kv_heads = self.q_heads // kv_groups
    self.kernel_size = kernel_size
    self.qkv_proj = Linear(d_model, self.d_head * (self.q_heads + 2*self.kv_heads), bias=False)
    self.out_proj = Linear(d_model, d_model, bias=False)

  def forward(self, x: FloatTensor) -> FloatTensor:
    qkv = self.qkv_proj(x)
    q, k, v = qkv.split((self.q_heads*self.d_head, *(self.kv_heads*self.d_head,)*2), -1)
    q, k, v = (rearrange(p, "n h w (nh e) -> n nh h w e", e=self.d_head) for p in (q,k,v))
    # TODO: should this be repeat_interleave? does it matter? I guess the best would be to match what Llama does?
    # TODO: could we give each of Q,K,V a "group" dim?
    #       Q  = (1, q_heads)
    #       KV = (kv_groups, kv_heads)
    #       could KV's group dim be created for free via expand()?
    # for now I'll just do a repeat, but it'd be really nice if we could use expand…
    k, v = (p.repeat(1, self.kv_groups, 1, 1, 1) for p in (k, v))
    q = q / self.d_head**.5
    qk = natten2dqk(q, k, self.kernel_size, 1)
    a = qk.softmax(dim=-1)
    x = natten2dav(a, v, self.kernel_size, 1)
    x = rearrange(x, "n nh h w e -> n h w (nh e)")
    x = self.out_proj(x)
    return x