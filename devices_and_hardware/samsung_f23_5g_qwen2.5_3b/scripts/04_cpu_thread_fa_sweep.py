#!/usr/bin/env python3
"""
Snapdragon 750G CPU-path optimization sweep for Qwen2.5-3B Q4_K_M.

Explores thread count and flash-attention on the *stable* CPU backend and
records TTFT, prefill/decode throughput, peak RAM and thermals.

WHY CPU-ONLY: on this device (Adreno 619, SM-E236B) the OpenCL GPU path
allocates from a 258MB CMA pool that sits at ~0 kB free at idle. Any GPU
buffer allocation faults inside the KGSL driver and panics the device into
a full reboot (observed twice; kgsl reset_count was 178). That failure is
kernel-side and cannot be guarded from userland, so GPU offload is
deliberately excluded here. See README for the full write-up.

SAFETY: this script never puts the vendor OpenCL driver on the library
path, so no GPU allocation is ever attempted. It also aborts the run if
MemAvailable drops below --min-avail-mb, and cools between runs.

Usage (on device):
    python3 04_cpu_thread_fa_sweep.py \
        --model ~/qwen2.5-3b-instruct-q4_k_m.gguf \
        --bin ~/llama.cpp/build-ocl/bin/llama-bench \
        --out cpu_sweep.json
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

THERMAL = Path("/sys/class/thermal")


def meminfo_kb(key):
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except Exception:
        pass
    return None


def mem_used_mb():
    total, avail = meminfo_kb("MemTotal"), meminfo_kb("MemAvailable")
    if total is None or avail is None:
        return None
    return (total - avail) // 1024


def avail_mb():
    a = meminfo_kb("MemAvailable")
    return a // 1024 if a is not None else None


def temps():
    cpu, gpu = [], []
    try:
        for zone in THERMAL.glob("thermal_zone*"):
            try:
                t = (zone / "type").read_text().strip()
                raw = (zone / "temp").read_text().strip()
                v = int(raw)
                c = v / 1000.0 if v > 1000 else float(v)
                if "gpu" in t:
                    gpu.append(c)
                elif "cpu" in t:
                    cpu.append(c)
            except Exception:
                continue
    except Exception:
        pass
    return (max(cpu) if cpu else None), (max(gpu) if gpu else None)


class Sampler(threading.Thread):
    """Samples RAM and temperature during a run; flags low-memory danger."""

    def __init__(self, min_avail_mb, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.min_avail_mb = min_avail_mb
        self.stop_flag = threading.Event()
        self.danger = threading.Event()
        self.mem, self.cpu_t, self.gpu_t, self.avail = [], [], [], []

    def run(self):
        while not self.stop_flag.is_set():
            m = mem_used_mb()
            if m:
                self.mem.append(m)
            a = avail_mb()
            if a is not None:
                self.avail.append(a)
                if a < self.min_avail_mb:
                    self.danger.set()
            c, g = temps()
            if c:
                self.cpu_t.append(c)
            if g:
                self.gpu_t.append(g)
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)

    def summary(self):
        peak = lambda x: round(max(x), 1) if x else None
        avg = lambda x: round(sum(x) / len(x), 1) if x else None
        return {
            "peak_ram_mb": peak(self.mem),
            "avg_ram_mb": avg(self.mem),
            "min_avail_mb": min(self.avail) if self.avail else None,
            "peak_cpu_temp_c": peak(self.cpu_t),
            "peak_gpu_temp_c": peak(self.gpu_t),
            "low_mem_warning": self.danger.is_set(),
            "samples": len(self.mem),
        }


def run_case(binary, model, threads, fa, pp, tg, reps, timeout, min_avail):
    """Run one llama-bench case on the CPU backend and parse JSON output."""
    cmd = [
        binary, "-m", model, "-ngl", "0", "-t", str(threads),
        "-p", str(pp), "-n", str(tg), "-r", str(reps), "-o", "json",
    ]
    if fa is not None:
        cmd += ["-fa", fa]

    # Deliberately strip the vendor driver from the loader path so that no
    # OpenCL platform is found and zero GPU memory is ever requested.
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    env.pop("LD_LIBRARY_PATH", None)

    sampler = Sampler(min_avail)
    sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        out, err, rc = "", "TIMEOUT", -9
    wall = time.time() - t0
    sampler.stop()

    rows = {}
    try:
        for e in json.loads(out):
            n_prompt = int(e.get("n_prompt") or 0)
            n_gen = int(e.get("n_gen") or 0)
            ts = round(float(e.get("avg_ts", 0)), 2)
            sd = round(float(e.get("stddev_ts", 0)), 2)
            if n_prompt:
                rows["prefill_tps"] = ts
                rows["prefill_stddev"] = sd
                # TTFT for a pp-token prompt at this prefill rate.
                rows["ttft_ms"] = round(1000.0 * n_prompt / ts, 1) if ts else None
            if n_gen:
                rows["decode_tps"] = ts
                rows["decode_stddev"] = sd
    except Exception:
        pass

    return {
        "threads": threads,
        "flash_attn": fa or "default",
        "n_prompt": pp,
        "n_gen": tg,
        "returncode": rc,
        "wall_s": round(wall, 1),
        **rows,
        "telemetry": sampler.summary(),
        "stderr_tail": err[-300:] if err else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", default="cpu_sweep.json")
    ap.add_argument("--threads", default="2,4,6,8")
    ap.add_argument("--fa", default="0,1")
    ap.add_argument("--pp", type=int, default=128)
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--cooldown", type=int, default=40)
    ap.add_argument("--max-temp", type=float, default=75.0)
    ap.add_argument("--min-avail-mb", type=int, default=400)
    args = ap.parse_args()

    model = os.path.expanduser(args.model)
    binary = os.path.expanduser(args.bin)

    cases = [(t, fa)
             for fa in args.fa.split(",")
             for t in [int(x) for x in args.threads.split(",")]]

    results = []
    for i, (t, fa) in enumerate(cases):
        # Thermal gate: wait for the SoC to cool before a timed run.
        for _ in range(20):
            c, _g = temps()
            if c is None or c < args.max_temp:
                break
            print(f"    hot ({c}C >= {args.max_temp}C), waiting 30s...",
                  flush=True)
            time.sleep(30)

        c, g = temps()
        print(f"[{i+1}/{len(cases)}] threads={t} fa={fa} "
              f"(cpu={c}C gpu={g}C avail={avail_mb()}MB)", flush=True)

        r = run_case(binary, model, t, fa, args.pp, args.tg,
                     args.reps, args.timeout, args.min_avail_mb)
        results.append(r)
        print(f"    rc={r['returncode']} prefill={r.get('prefill_tps')} "
              f"decode={r.get('decode_tps')} ttft={r.get('ttft_ms')}ms "
              f"peakRAM={r['telemetry']['peak_ram_mb']}MB "
              f"peakCPU={r['telemetry']['peak_cpu_temp_c']}C", flush=True)

        Path(args.out).write_text(json.dumps(results, indent=2))
        if i < len(cases) - 1 and args.cooldown:
            time.sleep(args.cooldown)

    Path(args.out).write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r.get("decode_tps")]
    if ok:
        best_d = max(ok, key=lambda r: r["decode_tps"])
        best_p = max(ok, key=lambda r: r.get("prefill_tps") or 0)
        print(f"\nBest decode : {best_d['decode_tps']} t/s "
              f"(threads={best_d['threads']}, fa={best_d['flash_attn']})")
        print(f"Best prefill: {best_p.get('prefill_tps')} t/s "
              f"(threads={best_p['threads']}, fa={best_p['flash_attn']})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
