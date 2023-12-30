import torch
from typing import List, Tuple
from torch import inference_mode, FloatTensor
from torch.nested import nested_tensor
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
y = torch.randn([1, canvas_len*2, canvas_len*2, d_model], device=device, dtype=dtype)

# default rtol of allclose
rtol=1e-5
# this is actually a *wildly* generous atol. the default would normally be 1e-08.
atol=5e-3

inputs: Tuple[FloatTensor, ...] = (x, y)
with inference_mode():
  outs_natt: List[FloatTensor] = [natten_block(input) for input in inputs]
  outs_hood: List[FloatTensor] = [hood_block(input) for input in inputs]
  for out_natt, out_hood in zip(outs_natt, outs_hood):
    assert out_natt.allclose(out_hood, rtol=1e-5, atol=5e-3), "assertion failure indicates implementations are not equivalent"
  print('Sanity-check passed; NATTEN matches pure-PyTorch for non-nested tensors')

  inputs_nest: FloatTensor = nested_tensor([x, y])
  outs_nest_natt: FloatTensor = natten_block(inputs_nest)
  # pure-Pytorch sdp actually doesn't support nested tensors, so let's just nest the per-sample results we got earlier
  for out_nest_natt, out_hood in zip(outs_nest_natt.unbind(), outs_hood):
    assert out_nest_natt.allclose(out_hood, rtol=1e-5, atol=5e-3), "assertion failure indicates NATTEN nested result is not equivalent to pure-PyTorch per-sample results"
  for out_nest_natt, out_natt in zip(outs_nest_natt.unbind(), outs_natt):
    assert out_nest_natt.allclose(out_natt), "assertion failure indicates NATTEN nested mode results aren't self-consistent with non-nested"

print(f'NATTEN output matched pure-PyTorch implementation to within atol={atol}, rtol={rtol}')