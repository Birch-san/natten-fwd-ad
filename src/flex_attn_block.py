from functools import lru_cache, partial
from torch import FloatTensor, IntTensor, BoolTensor
from torch.nn import Module, Linear
from einops import rearrange

from torch.nn.attention._flex_attention import _flex_attention as flex_attention
from src.flex_backports.flex_attention import create_block_mask#, flex_attention

@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
  block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
  return block_mask

def _natten_mask_mod(
  b: IntTensor,
  h: IntTensor,
  q_idx: IntTensor,
  kv_idx: IntTensor,
  canvas_width: int,
  kernel_width: int,
  kernel_height: int,
) -> BoolTensor:
  q_x, q_y = q_idx // canvas_width, q_idx % canvas_width
  kv_x, kv_y = kv_idx // canvas_width, kv_idx % canvas_width
  horizontal_mask = (q_x - kv_x).abs() <= kernel_width
  vertical_mask = (q_y - kv_y).abs() <= kernel_height
  return horizontal_mask & vertical_mask

class FlexAttnBlock(Module):
  def __init__(self, d_model: int, d_head: int, kernel_size: int):
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

  def forward(self, x: FloatTensor) -> FloatTensor:
    _, h, w, _ = x.shape
    qkv = self.qkv_proj(x)
    q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
    mask_mod = partial(_natten_mask_mod, canvas_width=w, kernel_width=self.kernel_size, kernel_height=self.kernel_size)
    block_mask = create_block_mask_cached(mask_mod, 1, 1, h*w, h*w, device=q.device)
    x = flex_attention(q, k, v, block_mask=block_mask)
    x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=h, w=w, e=self.d_head)
    x = self.out_proj(x)
    return x