from functools import lru_cache, partial
from torch import FloatTensor, IntTensor, BoolTensor
from torch.nn import Module, Linear
from einops import rearrange

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
  block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
  return block_mask

def _natten_mask_mod(
  b: IntTensor,
  h: IntTensor,
  q_idx: IntTensor,
  kv_idx: IntTensor,
  canvas_w: int,
  canvas_h: int,
  kernel_w: int,
  kernel_h: int,
) -> BoolTensor:
  q_x, q_y = q_idx // canvas_w, q_idx % canvas_w
  kv_x, kv_y = kv_idx // canvas_w, kv_idx % canvas_w
  hori_mask = (q_x.clamp(kernel_w//2, (canvas_w-1)-kernel_w//2) - kv_x).abs() <= kernel_w//2
  vert_mask = (q_y.clamp(kernel_h//2, (canvas_h-1)-kernel_h//2) - kv_y).abs() <= kernel_h//2
  return hori_mask & vert_mask

class FlexAttnBlock(Module):
  def __init__(self, d_model: int, d_head: int, kernel_size: int, verify_mask_parity=False):
    """
    Pure-PyTorch implementation of neighbourhood attention.
    Uses FlexAttention.
    """
    super().__init__()
    self.d_head = d_head
    self.n_heads = d_model // d_head
    self.kernel_size = kernel_size
    self.qkv_proj = Linear(d_model, d_model * 3, bias=False)
    self.out_proj = Linear(d_model, d_model, bias=False)
    self.verify_mask_parity = verify_mask_parity

  def forward(self, x: FloatTensor) -> FloatTensor:
    _, h, w, _ = x.shape
    mask_mod = partial(_natten_mask_mod, canvas_w=w, canvas_h=h, kernel_w=self.kernel_size, kernel_h=self.kernel_size)
    if self.verify_mask_parity:
      from .hood_attn_block import Dimensions, make_neighbourhood_mask
      from torch.nn.attention.flex_attention import create_mask
      flex_mask = create_mask(mask_mod, 1, 1, h*w, h*w, device=x.device)
      math_mask = make_neighbourhood_mask(
        Dimensions(self.kernel_size, self.kernel_size),
        Dimensions(h, w),
        flatten_to_1d=True,
        device=x.device
      )
      assert math_mask.unsqueeze(0).unsqueeze(0).allclose(flex_mask), "Flex attn mask does not match manually-made mask"
    qkv = self.qkv_proj(x)
    q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head).contiguous()
    block_mask = create_block_mask_cached(mask_mod, 1, 1, h*w, h*w, device=q.device)
    x = flex_attention(q, k, v, block_mask=block_mask)
    x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=h, w=w, e=self.d_head)
    x = self.out_proj(x)
    return x