# GPU Account Summation (CUDA)

A performance-oriented **CUDA C++** project (PV197 GPU programming, FI MUNI): compute per-client running account balances over a series of billing periods, together with per-period global totals, entirely on the GPU.

## What the kernel does

- Layout: each thread owns one client (column-major data), walking its periods in the main loop — `changes` in, running `account` balances out, per-period `sum` accumulated.
- The dependency chain within a client (balance `a` carries across periods) makes this a nontrivial parallel pattern: clients parallelize freely, but per-period totals must be reduced across them.

## Optimizations

- **Register blocking**: 16 periods per thread preloaded into registers (`#pragma unroll`), with an 8-way manually unrolled dependency chain to keep the running balance in registers.
- **Warp-level reduction**: per-period sums reduced inside a warp with `__shfl_down_sync` (butterfly/warp-sum), avoiding most shared-memory round-trips.
- **Shared-memory staging + `atomicAdd`**: one warp leader per warp accumulates into a `__shared__` period buffer, so only one atomic update per block hits global memory, then thread 0 flushes to the global `sum` array.

## Profiling

Nsight Compute capture files are included (`profile.ncu-rep`, `test0/test1.ncu-rep`); the kernel was iterated against these profiles (register pressure, atomic contention).

## Build

```sh
cd project
make        # nvcc; framework.cu + kernel.cu; CPU reference in kernel_CPU.C
```
