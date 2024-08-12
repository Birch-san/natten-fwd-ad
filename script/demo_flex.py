import torch
from torch import inference_mode, enable_grad
import torch.autograd.forward_ad as fwAD
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch.nn.attention.flex_attention import flex_attention
from src.natten_block import NattenBlock
from src.hood_attn_block import NeighbourhoodAttnBlock
from src.flex_attn_block import FlexAttnBlock

device=torch.device('cuda')
torch.set_default_device(device)
dtype=torch.bfloat16

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

seed=42
d_model=128
d_head=64
kernel_size=13
canvas_height = canvas_width = 32
torch.manual_seed(seed)
# fused kernel doesn't support fwAD
natten_block = NattenBlock(d_model, d_head=d_head, kernel_size=kernel_size, prefer_fused=False).to(device=device, dtype=dtype)
torch.manual_seed(seed)
hood_block = NeighbourhoodAttnBlock(d_model, d_head=d_head, kernel_size=kernel_size).to(device=device, dtype=dtype)
torch.manual_seed(seed)
flex_block = FlexAttnBlock(d_model, d_head=d_head, kernel_size=kernel_size).to(device=device, dtype=dtype)
torch.manual_seed(seed)

batch=2
x = torch.randn([batch, canvas_height, canvas_width, d_model], device=device, dtype=dtype)

with inference_mode():
  out_flex = flex_block(x)
  out_natt = natten_block(x)
  out_hood = hood_block(x)
# default rtol of allclose
rtol=1e-5
# this is actually a *wildly* generous atol. the default would normally be 1e-08.
atol=5e-3
assert out_flex.allclose(out_hood, rtol=1e-5, atol=5e-3), "assertion failure indicates Flex Attn is not equivalent to masked sdpa"
assert out_natt.allclose(out_hood, rtol=1e-5, atol=5e-3), "assertion failure indicates NATTEN is not equivalent to masked SDPA"
print(f'Flex and NATTEN outputs matched pure-PyTorch implementation to within atol={atol}, rtol={rtol}')

tangent = torch.randn([batch, canvas_height, canvas_width, d_model], device=device, dtype=dtype)
with fwAD.dual_level(), enable_grad(), sdpa_kernel(SDPBackend.MATH):
  dual_primal = fwAD.make_dual(x, tangent)
  out_natt = natten_block(dual_primal)
  out_natt_prime = fwAD.unpack_dual(out_natt).tangent
  out_hood = hood_block(dual_primal)
  out_hood_prime = fwAD.unpack_dual(out_hood).tangent
  out_flex = flex_block(dual_primal)
  out_flex_prime = fwAD.unpack_dual(out_flex).tangent
  # default rtol of allclose
  rtol=1e-5
  atol=1e-3
  assert out_natt_prime.allclose(out_hood_prime, rtol=rtol, atol=atol), "assertion failure indicates NATTEN fwAD is not equivalent to masked sdpa"
  assert out_flex_prime.allclose(out_hood_prime, rtol=rtol, atol=atol), "assertion failure indicates Flex Attn fwAD is not equivalent to masked sdpa"
  print(f'Flex and NATTEN fwAD outputs matched pure-PyTorch implementation to within atol={atol}, rtol={rtol}')