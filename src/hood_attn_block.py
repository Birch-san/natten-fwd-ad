import torch
from torch import FloatTensor, BoolTensor
from torch.nested import nested_tensor
from torch.nn import Module, Linear
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange
from typing import NamedTuple, List

class Dimensions(NamedTuple):
  height: int
  width: int

class QKVMask(NamedTuple):
  q: FloatTensor
  k: FloatTensor
  v: FloatTensor
  h: int
  w: int
  mask: BoolTensor

# by Katherine Crowson
def make_neighbourhood_mask(kernel_size: Dimensions, canvas_size: Dimensions, flatten_to_1d=False, device="cpu") -> BoolTensor:
  kernel_h, kernel_w = kernel_size
  canvas_h, canvas_w = canvas_size

  h_ramp = torch.arange(canvas_h, device=device)
  w_ramp = torch.arange(canvas_w, device=device)
  h_pos, w_pos = torch.meshgrid(h_ramp, w_ramp, indexing="ij")

  # Compute start_h and end_h
  start_h = torch.clamp(h_pos - kernel_h // 2, 0, canvas_h - kernel_h)[..., None, None]
  end_h = start_h + kernel_h

  # Compute start_w and end_w
  start_w = torch.clamp(w_pos - kernel_w // 2, 0, canvas_w - kernel_w)[..., None, None]
  end_w = start_w + kernel_w

  # Broadcast and create the mask
  h_range = h_ramp.reshape(1, 1, canvas_h, 1)
  w_range = w_ramp.reshape(1, 1, 1, canvas_w)
  mask = (h_range >= start_h) & (h_range < end_h) & (w_range >= start_w) & (w_range < end_w)

  if flatten_to_1d:
    mask = mask.view(canvas_h * canvas_w, canvas_h * canvas_w)

  return mask

class NeighbourhoodAttnBlock(Module):
  def __init__(self, d_model: int, d_head: int, kernel_size: int):
    """
    Pure-PyTorch implementation of neighbourhood attention.
    Uses global self-attention and a (very) complicated mask.
    Consequently it (probably) supports:
    - Mac
    - PyTorch Forward-Mode Autodiff
    - Nested tensors
    """
    super().__init__()
    self.d_head = d_head
    self.n_heads = d_model // d_head
    self.kernel_size = kernel_size
    self.qkv_proj = Linear(d_model, d_model * 3, bias=False)
    self.out_proj = Linear(d_model, d_model, bias=False)
  
  def _x_to_qkv_mask(self, x: FloatTensor) -> QKVMask:
    _, h, w, _ = x.shape
    qkv = self.qkv_proj(x)
    q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
    kernel_size=Dimensions(self.kernel_size, self.kernel_size)
    canvas_size=Dimensions(h, w)
    mask: BoolTensor = make_neighbourhood_mask(kernel_size, canvas_size, flatten_to_1d=True, device=x.device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return QKVMask(q, k, v, h, w, mask)
  
  def _attn_to_out(self, x: FloatTensor, h: int, w: int) -> FloatTensor:
    x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=h, w=w, e=self.d_head)
    x = self.out_proj(x)
    return x

  def forward(self, x: FloatTensor) -> FloatTensor:
    if x.is_nested:
      qkvms: List[QKVMask] = [self._x_to_qkv_mask(t) for t in x.unbind()]
      q = nested_tensor([qkvm.q for qkvm in qkvms])
      k = nested_tensor([qkvm.k for qkvm in qkvms])
      v = nested_tensor([qkvm.v for qkvm in qkvms])
      mask = nested_tensor([qkvm.mask for qkvm in qkvms])
      hs: List[int] = [qkvm.h for qkvm in qkvms]
      ws: List[int] = [qkvm.w for qkvm in qkvms]
    else:
      q, k, v, h, w, mask = self._x_to_qkv_mask(x)
    # note: fails for nested tensors, because nested tensor mask is not supported
    x = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if x.is_nested:
      outs: List[FloatTensor] = [self._attn_to_out(t, h, w) for t, h, w in zip(x.unbind(), hs, ws)]
      x = nested_tensor(outs)
    else:
      x = self._attn_to_out(x, h, w)
    return x