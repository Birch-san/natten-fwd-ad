import torch
from torch import inference_mode
# from src.natten_block import NattenBlock
from src.hood_attn_registers_block import NeighbourhoodAttnRegisterBlock

device=torch.device('cuda')
dtype=torch.bfloat16
seed=42
d_model=128
d_head=64
kernel_size=13
# torch.manual_seed(seed)
# natten_block = NattenBlock(d_model, d_head=d_head, kernel_size=kernel_size).to(device=device, dtype=dtype)
torch.manual_seed(seed)
hood_block = NeighbourhoodAttnRegisterBlock(d_model, d_head=d_head, kernel_size=kernel_size).to(device=device, dtype=dtype)

batch=2
canvas_len=32
register_count=16
x = torch.randn([batch, canvas_len, canvas_len, d_model], device=device, dtype=dtype)
registers = torch.randn([register_count, d_model], device=device, dtype=dtype)

with inference_mode():
  out_hood = hood_block(x, registers)
  # not yet supported by NATTEN
  # out_natt = natten_block(x, registers)
# assert out_natt.allclose(out_hood, rtol=1e-5, atol=5e-3), "assertion failure indicates implementations are not equivalent"
# print(f'NATTEN output matched pure-PyTorch implementation to within atol={atol}, rtol={rtol}')