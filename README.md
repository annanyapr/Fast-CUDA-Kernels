# GEMM Performance Comparison

The following table summarizes the performance (in GFLOPS) of different GEMM implementations on an NVIDIA GeForce RTX 4090 using 4096×4096 matrices:

| Variant                   | Elapsed Time (s) | Performance (GFLOPS) |
|---------------------------|------------------|----------------------|
| cuBLAS SGEMM              | 0.002887         | 47,611.8             |
| Memory Block Tiling       | 0.003694         | 37,201.0             |
| Memory Blocking           | 0.020849         | 6,592.0              |
| Memory Coalesce           | 0.028121         | 4,887.4              |
| Naive                     | 0.216326         | 635.3                |




# Softmax Kernel Performance

The following table shows the softmax kernel execution times (in milliseconds) measured on an NVIDIA GeForce RTX 4090:

| Variant                | Kernel Time (ms) |
|------------------------|------------------|
| PyTorch Kernel         | 0.7035           |
| Shuffle Instructions   | 0.90845          |
| Online Shared Memory   | 0.90854          |
| Online Naive           | 22.4709          |