import torch
from torch import inference_mode
from src.natten_registers_block import NattenRegistersBlock
from src.hood_attn_registers_block import NeighbourhoodAttnRegisterBlock

device=torch.device('cuda')
dtype=torch.bfloat16
seed=42
d_model=128
d_head=64
kernel_size=13
torch.manual_seed(seed)
natten_block = NattenRegistersBlock(d_model, d_head=d_head, kernel_size=kernel_size).to(device=device, dtype=dtype)
torch.manual_seed(seed)
hood_block = NeighbourhoodAttnRegisterBlock(d_model, d_head=d_head, kernel_size=kernel_size).to(device=device, dtype=dtype)

batch=2
canvas_len=32
register_count=16
x = torch.randn([batch, canvas_len, canvas_len, d_model], device=device, dtype=dtype)
registers = torch.randn([register_count, d_model], device=device, dtype=dtype)

with inference_mode():
  out_hood = hood_block(x, registers)
  out_natt = natten_block(x, registers)

# default rtol of allclose
rtol=1e-5
# this is actually a *wildly* generous atol. the default would normally be 1e-08.
atol=5e-3
assert out_natt.allclose(out_hood, rtol=rtol, atol=atol), "assertion failure indicates implementations are not equivalent"
print(f'NATTEN output matched pure-PyTorch implementation to within atol={atol}, rtol={rtol}')