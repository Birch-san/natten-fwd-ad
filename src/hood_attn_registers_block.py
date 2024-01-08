import torch
from torch import FloatTensor, BoolTensor, cat
from torch.nn import Module, Linear
from torch.nn.functional import scaled_dot_product_attention, linear
from einops import rearrange
from typing import NamedTuple, Optional

class Dimensions(NamedTuple):
  height: int
  width: int

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

class NeighbourhoodAttnRegisterBlock(Module):
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
    self.d_model = d_model
    self.qkv_proj = Linear(d_model, d_model * 3, bias=True)
    self.out_proj = Linear(d_model, d_model, bias=True)

  def forward(self, x: FloatTensor, registers: FloatTensor = None) -> FloatTensor:
    _, h, w, _ = x.shape
    qkv = self.qkv_proj(x)

    # this has nothing to do with registers, I'm just slicing a kv_proj out of qkv_proj
    kv_proj_weight: FloatTensor = self.qkv_proj.weight[self.d_model:, :]
    kv_proj_bias: Optional[FloatTensor] = None if self.qkv_proj.bias is None else self.qkv_proj.bias[self.d_model:]

    # project the registers through kv_proj (a separate kv_proj would be fine too)
    reg_kv: FloatTensor = linear(registers, kv_proj_weight, kv_proj_bias)
    # permute head channels into a batch dim, and unbind k and v from the fused kv
    reg_k, reg_v = rearrange(reg_kv, "r (t nh e) -> t 1 nh r e", t=2, e=self.d_head)
    # construct an all-True mask that says "attend to all registers"
    mask_reg = reg_k.new_ones(reg_k.size(-2), dtype=torch.bool)

    q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)

    # concatenate registers along the sequence dimension
    k_cat = cat([k, reg_k.expand(k.size(0), -1, -1, -1)], dim=-2)
    v_cat = cat([v, reg_v.expand(v.size(0), -1, -1, -1)], dim=-2)

    # make a neighbourhood mask for x
    kernel_size=Dimensions(self.kernel_size, self.kernel_size)
    canvas_size=Dimensions(h, w)
    mask: BoolTensor = make_neighbourhood_mask(kernel_size, canvas_size, flatten_to_1d=True, device=x.device)
    # query items should attend to their local neighbourhood *and* all registers
    mask_cat = cat([mask, mask_reg.expand(mask.size(0), -1)], dim=-1)
    # broadcast to all heads and batch items
    mask_cat = mask_cat.expand(1, 1, -1, -1)

    x = scaled_dot_product_attention(q, k_cat, v_cat, attn_mask=mask_cat)
    
    x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=h, w=w, e=self.d_head)
    x = self.out_proj(x)
    return x