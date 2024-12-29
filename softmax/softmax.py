import torch
import time

# Set up matrix dimensions
rows = 1024
columns = 1024 * 64  # 1024 times 64

# Generate random input tensor
matrix = torch.empty(rows, columns, device="cuda", dtype=torch.float32).uniform_(-5, 5)

# Warm-up to avoid startup overhead during benchmarking
for _ in range(10):
    _ = torch.nn.functional.softmax(matrix, dim=1)

# Benchmark the PyTorch softmax implementation
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
out = torch.nn.functional.softmax(matrix, dim=1)
end.record()
torch.cuda.synchronize()

print(start.elapsed_time(end), "ms (PyTorch kernel time)")