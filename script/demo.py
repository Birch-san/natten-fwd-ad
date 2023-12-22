import torch
from torch import inference_mode
from src.natten_block import NattenBlock

device=torch.device('cuda')
dtype=torch.bfloat16
torch.manual_seed(42)
d_model=128
natten_block = NattenBlock(d_model, d_head=64, kernel_size=13).to(device=device, dtype=dtype)

canvas_len=32
x = torch.randn([1, canvas_len, canvas_len, d_model], device=device, dtype=dtype)

with inference_mode():
  out = natten_block(x)
  pass