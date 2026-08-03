# Adreno 619 GPU Offload — Failed Experiment Write-Up

**Device:** Samsung Galaxy F23 5G (SM-E236B) · Snapdragon 750G (`lito`) · Adreno 619
**OS:** Android 14 (SDK 34), kernel 4.19.152, unrooted
**Model:** Qwen2.5-3B-Instruct Q4_K_M (1.95 GiB, 3.40B params)
**Engine:** llama.cpp `6bdd77f13` OpenCL backend
**Date:** 2026-08-04
**Outcome:** ❌ GPU offload not viable on this device. Root cause is a kernel-level
memory-reservation limit, not a build or configuration error.

---

## TL;DR

We successfully built llama.cpp's OpenCL backend in Termux and got it to **detect and
address the Adreno 619 as a real compute device**. Inference offload still fails: every
attempt to allocate a GPU buffer panics the device into a full reboot.

The cause is **CMA (Contiguous Memory Allocator) exhaustion**. The Adreno 619 has no
dedicated VRAM. It allocates from a fixed 252 MiB contiguous region reserved at boot,
which on this device is essentially fully consumed by the display/camera stack before
Termux ever runs. Requesting GPU memory from an empty CMA pool faults inside the KGSL
driver, and because that driver owns the display pipeline, the failure escalates to a
kernel panic and reboot rather than returning a catchable error to userspace.

This is **not fixable from userspace**. It requires root to enlarge the CMA reservation
via kernel cmdline, or different hardware.

---

## Why this looked promising

Published guidance (and the Adreno-specific kernel work upstream in llama.cpp, e.g.
PR #25810 tuning MoE dp4a tiles for Adreno) suggests Snapdragon GPU offload should give
a meaningful speedup over CPU inference — commonly quoted as 15–22 tok/s for a 3B model
versus our ~4.9 tok/s CPU baseline.

The hardware genuinely supports it:

```
Platform Name     QUALCOMM Snapdragon(TM)
Device Name       QUALCOMM Adreno(TM)
Device Version    OpenCL 2.0 Adreno(TM) 619
Device Type       GPU
Max work group    1024
Local memory      32 KiB
Global memory     2845562880 (2.65 GiB)   <-- misleading, see "The trap" below
```

The driver is present, current, and functional. `clinfo` enumerates it correctly.
The problem is not capability — it is memory *reservation*.

---

## Three real problems solved before hitting the wall

These are genuine, reusable Termux/Android findings. Anyone attempting Adreno OpenCL in
Termux will hit all three.

### 1. Android linker namespace blocks `/vendor` libraries

Termux cannot `dlopen()` a `/vendor` library by absolute path. The Android linker
enforces namespace isolation:

```
E linker: library "/vendor/lib64/libOpenCL.so" needed or dlopened by
"/data/data/com.termux/files/usr/lib/libOpenCL.so" is not accessible for the
namespace: [name="(default)", ... permitted_paths="/system/lib64/drm:...:/data:..."]
```

`/vendor/lib64` is absent from `permitted_paths`; `/data` is present.

**Fix:** copy the vendor driver into `/data` and point the ICD at the copy, while
supplying `/vendor/lib64` on the library path so the copy's own dependencies
(`libgsl.so`, `libCB.so`, `libllvm-qcom.so`, `libadreno_utils.so`) still resolve.

```bash
mkdir -p $HOME/adreno_cl
cp /vendor/lib64/libOpenCL.so $HOME/adreno_cl/libadrenocl.so
mkdir -p $PREFIX/etc/OpenCL/vendors
echo "$HOME/adreno_cl/libadrenocl.so" > $PREFIX/etc/OpenCL/vendors/adreno.icd
```

Note the vendor `libOpenCL.so` is only a thin Adreno-CB wrapper; the real driver work
happens in the libraries it pulls in, which is why the vendor path is still required.

### 2. `LD_LIBRARY_PATH=/vendor/lib64` breaks every Termux binary

The obvious approach — exporting the vendor path globally — poisons the loader for all
Termux executables, because `/vendor/lib64/libc++.so` shadows Termux's own C++ runtime:

```
CANNOT LINK EXECUTABLE "cmake": cannot locate symbol "_ZNSt6__ndk1..." referenced by
".../bin/cmake"
```

This breaks `cmake`, `clang`, and `llama-bench` alike. Path *ordering* does not reliably
fix it, since the vendor `libc++` still wins symbol resolution for some binaries.

**Fix:** force Termux's C++ runtime to load first with `LD_PRELOAD`, and scope the
vendor path to the run only — never export it into the build environment.

```bash
LD_PRELOAD=$PREFIX/lib/libc++_shared.so \
LD_LIBRARY_PATH=/vendor/lib64 \
./llama-bench --list-devices
```

### 3. Build targets OpenCL 3.0; the Adreno 619 driver is OpenCL 2.0

Default build fails to link:

```
cannot locate symbol "clCreateBufferWithProperties" referenced by libggml-opencl.so
```

`clCreateBufferWithProperties` is an OpenCL 3.0 API. The Adreno 619 driver is 2.0 and
does not export it. Upstream already guards this behind a version macro:

```c
#if GGML_OPENCL_TARGET_VERSION >= 300
    // clCreateBufferWithProperties and cl_mem_properties are OpenCL 3.0. Drivers older
    // than that do not export the symbol, so a build targeting them fails to link.
```

**Fix:** a supported CMake option, no patching required.

```bash
cmake -B build-ocl -DGGML_OPENCL=ON -DGGML_OPENCL_TARGET_VERSION=200 ...
```

### Result: the GPU is detected

```
ggml_opencl: selected platform: 'QUALCOMM Snapdragon(TM)'
ggml_opencl: device: 'QUALCOMM Adreno(TM) (OpenCL 2.0 Adreno(TM) 619)'
Available devices:
  GPUOpenCL: QUALCOMM Adreno(TM) (2713 MiB, 1689 MiB free)
```

Everything up to this point works. The wiring is correct.

---

## The actual wall: CMA exhaustion

### The trap in that output

`2713 MiB, 1689 MiB free` is **not real**. That figure is the driver's theoretical
addressable limit, not memory it can actually obtain. The real budget is the CMA pool:

```
$ grep -E "MemTotal|CmaTotal|CmaFree" /proc/meminfo
MemTotal:        5557740 kB     # 5.3 GiB system RAM
CmaTotal:         258048 kB     # 252 MiB reserved contiguous region
CmaFree:               0 kB     # <-- at idle, three samples 5s apart: 0, 24, 0 kB
```

`CmaFree` sits at **0–40 kB at idle**, before any inference starts. The display and
camera stacks have already claimed the pool. So although the phone reports gigabytes of
free RAM, the memory an integrated GPU is *allowed* to use is fully spoken for.

This is the answer to the obvious question — "why does it die when 6 GB is free?"
General-purpose free RAM and DMA-capable contiguous GPU memory are different resources.
The GPU cannot use the former.

### Failure mechanism

When llama.cpp requests a GPU buffer:

1. KGSL attempts a CMA allocation from a pool with ~0 bytes free.
2. The allocation fails inside kernel context, in a driver that owns the display pipeline.
3. The GPU is reset. When recovery fails, the panic takes the device down.

Evidence — `reset_count` is cumulative GPU reset/recovery events:

| Checkpoint | `reset_count` |
|---|---|
| Before our GPU attempts | 178 |
| After the offload attempts | **209** |

31 additional GPU resets logged across our runs. Two of them escalated to full device
reboots (observed directly: SSH dropped, `uptime` reset, hotspot re-IP'd each time).

### The decisive detail: `-ngl 4` fails exactly like `-ngl 99`

We first tried `-ngl 99` (all 36 layers). Reboot. The intuitive read is "asked for too
much, lower it" — so we retried with `-ngl 4`, just 4 layers, a few tens of MB.

**It rebooted identically.**

That is the diagnostic result. The failure is not proportional to allocation size; it
occurs at the *first* CMA allocation, because the pool is empty rather than merely small.
No `-ngl` value avoids it. There is no safe layer count on this device.

### Control experiment

Running the same binary with the vendor driver *removed* from the library path means no
OpenCL platform is found, so no GPU allocation is ever attempted:

```
ggml_opencl: platform IDs not available.
| qwen2 3B Q4_K - Medium | 1.95 GiB | 3.40 B | OpenCL | 0 | 4 | pp32 | 14.37 ± 0.00 |
| qwen2 3B Q4_K - Medium | 1.95 GiB | 3.40 B | OpenCL | 0 | 4 | tg16 |  5.09 ± 0.00 |
```

Completely stable, no reboot. Combined with the reset counter, this isolates GPU buffer
allocation as the sole trigger. Same binary, same model, same thermal state — the only
variable is whether a GPU allocation is attempted.

---

## Vulkan: also a dead end (different reason)

Vulkan was the recommended first choice. It fails earlier and more cheaply.

Termux's `vulkan-loader` only registers a software rasterizer:

```
$ ls $PREFIX/share/vulkan/icd.d/
lvp_icd.aarch64.json          # lavapipe — CPU, not the GPU

$ vulkaninfo --summary
deviceName = llvmpipe (LLVM 21.1.8, 128 bits)
deviceType = PHYSICAL_DEVICE_TYPE_CPU
```

The real Adreno Vulkan driver exists at `/vendor/lib64/hw/vulkan.adreno.so` (1.9 MB) but
is unreachable from Termux for the same namespace reason as problem #1, and it has no ICD
manifest to register it.

Critically, **benchmarking `llvmpipe` would have been worthless** — a CPU software
rasterizer emulating a GPU is strictly slower than the native CPU backend. Any "Vulkan"
number produced this way would have been a fabricated speedup measuring the wrong thing.
Worth flagging: it silently *looks* like it works.

Even if the ICD were wired up, Vulkan allocates from the same exhausted CMA pool, so the
outcome would match OpenCL.

---

## Reproducing the build (for a device with CMA headroom)

The build recipe is sound and should work on hardware with an adequate CMA reservation.

```bash
pkg install -y cmake clang opencl-headers ocl-icd clinfo

# Register the vendor driver via a /data copy (namespace workaround)
mkdir -p $HOME/adreno_cl $PREFIX/etc/OpenCL/vendors
cp /vendor/lib64/libOpenCL.so $HOME/adreno_cl/libadrenocl.so
echo "$HOME/adreno_cl/libadrenocl.so" > $PREFIX/etc/OpenCL/vendors/adreno.icd

# Verify the GPU is visible before building
LD_LIBRARY_PATH=/vendor/lib64 clinfo | grep -E "Device Name|Device Version"

# Configure — do NOT export LD_LIBRARY_PATH here, it breaks cmake/clang
cd ~/llama.cpp
cmake -B build-ocl -DGGML_OPENCL=ON -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENCL_TARGET_VERSION=200 \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DOpenCL_LIBRARY=/vendor/lib64/libOpenCL.so \
  -DOpenCL_INCLUDE_DIR=$PREFIX/include
cmake --build build-ocl -j8

# Run — vendor path scoped to the command only, Termux libc++ preloaded
LD_PRELOAD=$PREFIX/lib/libc++_shared.so LD_LIBRARY_PATH=/vendor/lib64 \
  ./build-ocl/bin/llama-bench --list-devices
```

### Check CMA headroom before attempting offload

**Do this first.** It is the difference between a benchmark and a reboot loop:

```bash
grep -E "CmaTotal|CmaFree" /proc/meminfo
```

If `CmaFree` is near zero, stop — GPU offload will panic the device. You want
`CmaFree` comfortably above the size of the layers you intend to offload.

---

## Conclusions

1. **Adreno 619 GPU offload is not viable on the Galaxy F23 5G.** The 252 MiB CMA pool is
   fully consumed by the display stack at idle. The limit is memory reservation, not
   compute capability, driver version, or build configuration.

2. **The 15–22 tok/s projection is unreachable on this class of device.** Those figures
   assume the GPU can obtain working memory. On an entry-level SoC where CMA is sized for
   the display pipeline alone, it cannot.

3. **Failure mode is severe, not graceful.** A userspace allocation request produces a
   kernel panic and full reboot, because KGSL owns the display pipeline. There is no
   errno to catch and no way to guard it from Termux — which makes the CMA pre-check
   above mandatory rather than optional.

4. **`llama.cpp`'s Adreno support is genuinely good.** The OpenCL backend, its Adreno
   kernels, and the `GGML_OPENCL_TARGET_VERSION` escape hatch all worked as designed.
   Nothing here is a llama.cpp defect.

5. **Where it *would* work:** a device with a larger CMA carveout, or a rooted device
   where `cma=512M` (or larger) can be set on the kernel cmdline. Flagships with 8 GB+
   typically reserve substantially more.

### Silver lining

Ruling out the GPU pushed the work to the CPU path, which produced a genuinely useful
result: **prefill and decode want opposite thread counts** on this SoC. Prefill scales
with threads (compute-bound), decode peaks at 2 threads and degrades beyond that
(memory-bandwidth-bound). The previous `-t 4` default was the worst of both. See the
device README for the full table — best decode improved ~13% over baseline for free.

A negative result that costs two reboots and yields a reusable Termux/Android build
recipe plus a hard hardware limit is a fair trade. Fun experiment.
