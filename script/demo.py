import torch
from torch import inference_mode, enable_grad
import torch.autograd.forward_ad as fwAD
from torch.backends.cuda import sdp_kernel
from src.natten_block import NattenBlock
from src.hood_attn_block import NeighbourhoodAttnBlock

device=torch.device('cuda')
dtype=torch.bfloat16
seed=42
d_model=384
d_head=64
kernel_size=13
torch.manual_seed(seed)
natten_block = NattenBlock(d_model, d_head=d_head, kv_groups=2, kernel_size=kernel_size).to(device=device, dtype=dtype)
torch.manual_seed(seed)
hood_block = NeighbourhoodAttnBlock(d_model, d_head=d_head, kv_groups=2, kernel_size=kernel_size).to(device=device, dtype=dtype)

batch=2
canvas_len=32
x = torch.randn([batch, canvas_len, canvas_len, d_model], device=device, dtype=dtype)

with inference_mode():
  out_natt = natten_block(x)
  out_hood = hood_block(x)
# default rtol of allclose
rtol=1e-5
# this is actually a *wildly* generous atol. the default would normally be 1e-08.
atol=5e-3
assert out_natt.allclose(out_hood, rtol=1e-5, atol=5e-3), "assertion failure indicates implementations are not equivalent"
print(f'NATTEN output matched pure-PyTorch implementation to within atol={atol}, rtol={rtol}')

tangent = torch.randn([batch, canvas_len, canvas_len, d_model], device=device, dtype=dtype)
with fwAD.dual_level(), enable_grad(), sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
  dual_primal = fwAD.make_dual(x, tangent)
  out_natt = natten_block(dual_primal)
  out_natt_prime = fwAD.unpack_dual(out_natt).tangent
  out_hood = hood_block(dual_primal)
  out_hood_prime = fwAD.unpack_dual(out_hood).tangent
  # default rtol of allclose
  rtol=1e-5
  atol=1e-3
  assert out_natt_prime.allclose(out_hood_prime, rtol=rtol, atol=atol), "assertion failure indicates fwAD implementations are not equivalent"
  print(f'NATTEN fwAD output matched pure-PyTorch implementation to within atol={atol}, rtol={rtol}')