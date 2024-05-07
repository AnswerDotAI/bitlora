import os
import torch
from torch.utils.cpp_extension import load_inline, load

# TODO(avh): don't hard code arch list
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"

sources = ["bindings.cu"]
try:
    module = load(
        name="qdora", sources=sources, verbose=True
    )
except Exception as e:
    print(f"Exception while loading {sources}: {e}")


print("Initializing values")

M = 64
K = 256
N = 64
GS = 8
R = 32 # lora rank
NGROUPS = M * K // GS

# 10 values per 32 bits
# torch doesn't have a uint32_t type, so we use int32_t for size
# don't operate on the values except as a bit stream.
wq = torch.randint(100, (M, ((K + 9) // 10)), dtype=torch.int32).cuda()
z = torch.randn(M * K // GS).cuda()
s = torch.randn(M * K // GS).cuda()
x = torch.randn(M * K).cuda()
out = torch.zeros(M * N).cuda()
loraA = torch.randn(M * R).cuda()
loraB = torch.randn(R * N).cuda()
doraScale = 1.5;

print("Invoking kernel...")

module.qdora(wq, z, s, x, out, M, K, N, GS, loraA, loraB, R, doraScale)

print(f"Result: {out}")
