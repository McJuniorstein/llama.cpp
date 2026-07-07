# MoE offload tuning — verified patches + what actually works

This branch (`fleet/moe-offload`) carries a small set of patches for **MoE models whose
experts are offloaded to system RAM** (`--n-cpu-moe`), plus the results of independently
verifying them on two machines. If your model fits entirely in VRAM, nothing here helps you.

The starting point was [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp)
(`fable5/host-register` and `fable5/prefetch-experts`, from the video
["I Asked Claude Fable 5 to Improve llama.cpp"](https://youtu.be/VytSYCDhWQ0)). We
cherry-picked his three commits, verified every claim on our own hardware, found that the
headline number does not reproduce as shipped, fixed why, and kept what works. Full credit
to thecodacus for the prefetch idea — it's real and it survives adversarial testing.

## TL;DR — the recipe

```bash
GGML_SCHED_PREFETCH_EXPERTS=1 ./llama-server \
    -m model.gguf -ngl 99 --n-cpu-moe <N> --no-mmap -ub 2048 ...
```

- `--no-mmap` is the big one on some systems (**+40% prefill on machine A, +5% on machine B**)
  and it's **mainline behavior, no patch needed**: without mmap, CPU-kept experts are
  allocated in pinned `CUDA_Host` (cudaMallocHost) memory, so PCIe uploads run at full
  speed. Cost: slower model load, higher transient RAM use.
- `GGML_SCHED_PREFETCH_EXPERTS=1` enables the prefetch patch from this branch:
  **+4–8% prefill** on top of `--no-mmap`, output token-identical, decode unaffected.
- `GGML_CUDA_REGISTER_HOST=1` (the pinning patch): **likely a no-op on your machine** —
  see below. Harmless to leave off.

## Test setup

| | Machine A | Machine B |
|---|---|---|
| GPU | RTX 3090 24GB (PCIe 4.0) | RTX 4080 Super 16GB (PCIe 4.0 x8*) |
| CPU | i7-12700KF (AVX2) | Ryzen 7 9800X3D (AVX-512 VNNI/BF16) |
| RAM | 128GB DDR4 | 128GB DDR5 |
| OS / CUDA | Ubuntu 24.04, CUDA 12.9, driver 580 | Fedora 43, CUDA 13.2, driver 580 |

Model: Qwen3-Coder-30B-A3B Q4_K_M (128 experts, ~3B active), `llama-bench -ngl 99 -ub 2048`,
pp2048 / tg32, 3 repetitions. `-ncmoe 99` = all experts in RAM (worst case);
`-ncmoe 24` = half offloaded (realistic "model slightly too big for VRAM" case).

## Results (prefill t/s / decode t/s)

| Config | Machine A (3090) | Machine B (4080S) |
|---|---|---|
| mmap default, ncmoe 99 | 998 / 26.2 | 1487 / 34.8 |
| mmap + both env vars (video config) | **955 / 26.0 (regression!)** | 1543 / 35.7 |
| no-mmap, ncmoe 99 | 1401 / 26.3 | 1561 / 37.3 |
| no-mmap + prefetch, ncmoe 99 | 1459 / 26.2 | 1576 / 37.9 |
| no-mmap, ncmoe 24 | 2157 / 45.8 | 2481 / 66.3 |
| **no-mmap + prefetch, ncmoe 24** | **2333 / 45.9** | **2631 / 66.7** |

Notes:
- Decode is untouched by all of this — it's bound by CPU/RAM reading the experts.
  (Machine B's +45% decode over A is DDR5 + AVX-512, not patches.)
- The original video reported +64.5% total on a 3060 12GB. On our machines the
  as-shipped env vars were **-4% to +4%**. Most of the video's gain corresponds to what
  mainline `--no-mmap` already does; see next section.

## Round 2: `-ub` is the biggest lever, and it multiplies the prefetch patch

Two findings from continued tuning that dwarf everything above:

**1. Offloaded-MoE prefill scales almost linearly with ubatch.** The same expert bytes
stream over PCIe per pass regardless of batch size, so doubling `-ub` roughly doubles
prefill throughput. Machine B, `-ncmoe 24`, pp4096:

| -ub | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| prefill t/s | 725 | 1396 | 2575 | **4309** |

Compute buffers grow with ubatch — on a 16GB card, ub 8192 only fits at full expert
offload (`-ncmoe 99`), where it reached **5286 t/s** (pp8192).

**2. The prefetch patch and big ubatch are synergistic, not additive.** At ub 2048 the
prefetch overlap is worth +4–8%. At ub 8192 (machine B, ncmoe 99, pp8192):
prefetch OFF = 3302 t/s, prefetch ON = **5286 t/s (+60%)**. Larger batches mean more
compute per layer to hide under the copies. Use them together.

**VRAM is a dial, not a setting:** spend it on ubatch buffers (prefill) or on resident
expert layers (decode). Decode doesn't care about ubatch; prefill barely cares about
resident layers once ubatch is large. Tune per workload.

## Bonus: gpt-oss-120b on a 24GB RTX 3090 (128GB DDR4 host)

Same recipe applied to a model ~2.5× bigger than VRAM (MXFP4, 59GiB), machine A:

| -ncmoe | -ub | prefill t/s | decode t/s | VRAM |
|---|---|---|---|---|
| 99 (all experts in RAM) | 8192 | 1362 | 16.1 | ~7G |
| 30 | 4096 | 855 | 18.9 | — |
| **30** | **8192** | **1506** | **19.1** | 15.5G |
| 26 | 8192 | — | 21.8 | 22.0G |

A 117B-parameter MoE at 1500 t/s prefill / 19–22 t/s decode on a single consumer GPU.
For chat-heavy use, more resident layers (`-ncmoe 26`); for RAG/long-context, keep
VRAM headroom and max ubatch.

## Why the pinning patch probably does nothing on your machine

The `GGML_CUDA_REGISTER_HOST` patch pins mmap'd model pages with `cudaHostRegister` so
PCIe copies run at pinned speed while keeping mmap's instant load. Two hard blockers we
hit on **both** test machines (verified with standalone CUDA programs, not just llama.cpp):

1. `cudaHostRegister` on **file-backed mmap pages** fails (`operation not supported` /
   `invalid argument`) on our driver/kernel combos. It evidently works on some setups
   (the video's), but don't assume yours.
2. The `cudaHostRegisterReadOnly` flag used by the (previously dormant) upstream helper
   requires `cudaDevAttrHostRegisterReadOnlySupported`, which consumer GPUs we tested
   don't report — so registration fails **even for regular malloc memory**.

Because failures are logged at debug level only, the patch fails *silently* and you
benchmark a placebo. That's the trap we fell into first, and likely many others will too.

Our commits on this branch fix what's fixable:
- retry registration without the ReadOnly flag (`ggml-cuda.cu`),
- pin malloc-backed weight buffers in the no-mmap path, collected correctly across
  per-context loader calls (`llama-model-loader.*`),
- skip buffers that are already pinned (`CUDA_Host`) instead of failing on them every load.

After all that: on systems where `--no-mmap` already gives you pinned experts, explicit
pinning adds ~0%. We kept the fixes because they make the feature honest — if it can't
pin, that's now visible and recoverable — but temper your expectations.

## The patch that IS worth it: expert prefetch overlap

`GGML_SCHED_PREFETCH_EXPERTS=1` (thecodacus's idea, with his follow-up fix for slot
sizing and a use-after-free). During large-batch prefill, nearly every expert gets
routed to anyway — so instead of computing attention, waiting for router IDs, and only
then uploading experts (GPU idle during every copy), a second stream uploads the whole
expert tensors for the next layer while the current one computes. Three staging slots
(gate/up/down) give one full layer of lookahead; more slots measured no better.

Verified: +4–8% prefill on both machines, generation byte-identical, no decode impact,
graceful fallback when device memory for slots runs out.

## Check your PCIe link before tuning software

This workload is bound by host→GPU transfers, so check what your GPU actually trains at:

```bash
cat /sys/bus/pci/devices/0000:$(lspci | grep -i 'vga.*nvidia' | cut -d' ' -f1)/current_link_width
```

Machine B's 4080S was silently running **x8** — on AM5 boards (X870E at least), the
CPU-attached M.2 slots are carved out of the GPU's 16 lanes, so two populated M.2 slots
cost the GPU half its bandwidth: 8+4+4. Moving NVMe drives to chipset M.2 slots restores
x16. That single hardware change is likely worth more than every patch on this branch
combined for offloaded-MoE prefill.

## Reproducing

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j --target llama-bench llama-server

# baseline / tuned:
./build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 24 -ub 2048 -b 2048 -p 2048 -n 32 -r 3 -mmp 0
GGML_SCHED_PREFETCH_EXPERTS=1 ./build/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 24 -ub 2048 -b 2048 -p 2048 -n 32 -r 3 -mmp 0
```

Verify output identity with `llama-completion` at `--temp 0 --seed <fixed>` with and
without the env vars.

---

*Patches from [thecodacus](https://github.com/thecodacus/llama.cpp) (see his video for
the origin story); verification, fixes, and this writeup by the fork owner with
Claude (Fable 5). AI involvement is disclosed in each commit. This branch is a private
experiment made public — it is not affiliated with upstream llama.cpp, and per upstream's
contribution policy these AI-assisted patches are not submitted as PRs.*
