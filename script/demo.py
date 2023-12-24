import torch
from torch import inference_mode
from src.natten_block import NattenBlock
from src.hood_attn_block import NeighbourhoodAttnBlock

device=torch.device('cuda')
dtype=torch.bfloat16
seed=42
d_model=128
d_head=64
kernel_size=13
torch.manual_seed(seed)
natten_block = NattenBlock(d_model, d_head=d_head, kernel_size=kernel_size).to(device=device, dtype=dtype)
torch.manual_seed(seed)
hood_block = NeighbourhoodAttnBlock(d_model, d_head=d_head, kernel_size=kernel_size).to(device=device, dtype=dtype)

canvas_len=32
x = torch.randn([1, canvas_len, canvas_len, d_model], device=device, dtype=dtype)

with inference_mode():
  out = natten_block(x)
  out2 = hood_block(x)
  pass