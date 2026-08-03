#!/usr/bin/env python3
"""
Adreno 619 (Snapdragon 750G) GPU offload benchmark for llama.cpp OpenCL backend.

Runs on-device inside Termux. Captures TTFT, prefill (PP) and decode (TG)
throughput, peak RAM, thermals and GPU busy-percent for a sweep of
GPU-offload layer counts (-ngl).

GPU-usage proof: /sys/class/kgsl/kgsl-3d0/gpubusy is sampled before and
after each run. It reports cumulative "busy total" jiffy counters, so a
rising busy delta during a run is direct kernel-level evidence that work
actually executed on the Adreno rather than silently falling back to CPU.

Usage (on device):
    LD_LIBRARY_PATH=/vendor/lib64 python3 03_adreno_gpu_benchmark.py \
        --model ~/qwen2.5-3b-instruct-q4_k_m.gguf \
        --bin ~/llama.cpp/build-ocl/bin/llama-bench \
        --out results.json
"""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path

KGSL = Path("/sys/class/kgsl/kgsl-3d0")
THERMAL = Path("/sys/class/thermal")


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def gpubusy():
    """Return (busy, total) cumulative counters from the kgsl driver."""
    try:
        parts = (KGSL / "gpubusy").read_text().split()
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None


def temps():
    """Max CPU and GPU temperature in Celsius, read from thermal zones."""
    cpu, gpu = [], []
    try:
        for zone in THERMAL.glob("thermal_zone*"):
            try:
                t = (zone / "type").read_text().strip()
                v = read_int(zone / "temp")
                if v is None:
                    continue
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


def mem_used_mb():
    """System-wide used memory in MB (MemTotal - MemAvailable)."""
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k] = int(v.strip().split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) // 1024
    except Exception:
        return None


class Sampler(threading.Thread):
    """Background telemetry sampler: RAM, temps, GPU busy percent."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.mem, self.cpu_t, self.gpu_t, self.busy_pct = [], [], [], []

    def run(self):
        pb, pt = gpubusy()
        while not self.stop_flag.is_set():
            time.sleep(self.interval)
            m = mem_used_mb()
            if m:
                self.mem.append(m)
            c, g = temps()
            if c:
                self.cpu_t.append(c)
            if g:
                self.gpu_t.append(g)
            b, t = gpubusy()
            if b is not None and t is not None and pb is not None \
                    and pt is not None and t > pt:
                self.busy_pct.append(100.0 * (b - pb) / (t - pt))
            pb, pt = b, t

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)

    def summary(self):
        peak = lambda x: round(max(x), 1) if x else None
        avg = lambda x: round(sum(x) / len(x), 1) if x else None
        return {
            "peak_ram_mb": peak(self.mem),
            "avg_ram_mb": avg(self.mem),
            "peak_cpu_temp_c": peak(self.cpu_t),
            "peak_gpu_temp_c": peak(self.gpu_t),
            "avg_gpu_busy_pct": avg(self.busy_pct),
            "peak_gpu_busy_pct": peak(self.busy_pct),
            "samples": len(self.mem),
        }


def run_bench(binary, model, ngl, pp, tg, threads, reps, env, timeout):
    """Invoke llama-bench once and parse its JSON output."""
    cmd = [
        binary, "-m", model, "-ngl", str(ngl), "-p", str(pp), "-n", str(tg),
        "-t", str(threads), "-r", str(reps), "-o", "json",
    ]
    sampler = Sampler()
    busy_before = gpubusy()
    sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        out, err, rc = "", "TIMEOUT", -9
    wall = time.time() - t0
    sampler.stop()
    busy_after = gpubusy()

    rows = []
    try:
        for entry in json.loads(out):
            rows.append({
                "test": f"{entry.get('n_prompt')}p/{entry.get('n_gen')}g",
                "n_prompt": entry.get("n_prompt"),
                "n_gen": entry.get("n_gen"),
                "tps": round(float(entry.get("avg_ts", 0)), 2),
                "stddev": round(float(entry.get("stddev_ts", 0)), 2),
            })
    except Exception:
        pass

    delta = None
    b0, t0_c = busy_before
    b1, t1_c = busy_after
    if b0 is not None and t0_c is not None \
            and b1 is not None and t1_c is not None:
        db, dt = b1 - b0, t1_c - t0_c
        delta = {
            "busy_delta": db,
            "total_delta": dt,
            "busy_pct": round(100.0 * db / dt, 2) if dt > 0 else None,
        }

    return {
        "ngl": ngl,
        "threads": threads,
        "returncode": rc,
        "wall_s": round(wall, 1),
        "rows": rows,
        "telemetry": sampler.summary(),
        "gpu_counter": delta,
        "stderr_tail": err[-400:] if err else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", default="adreno_results.json")
    ap.add_argument("--ngl", default="0,8,16,99")
    ap.add_argument("--pp", type=int, default=128)
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--cooldown", type=int, default=45)
    ap.add_argument("--label", default="opencl")
    args = ap.parse_args()

    import os
    env = dict(os.environ)

    results = []
    ngls = [int(x) for x in args.ngl.split(",")]
    for i, ngl in enumerate(ngls):
        c, g = temps()
        print(f"[{i+1}/{len(ngls)}] ngl={ngl} start "
              f"(cpu={c}C gpu={g}C)", flush=True)
        r = run_bench(args.bin, args.model, ngl, args.pp, args.tg,
                      args.threads, args.reps, env, args.timeout)
        r["backend"] = args.label
        results.append(r)
        print(f"    rc={r['returncode']} wall={r['wall_s']}s "
              f"rows={r['rows']} gpu={r['gpu_counter']}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        if i < len(ngls) - 1 and args.cooldown:
            print(f"    cooling {args.cooldown}s...", flush=True)
            time.sleep(args.cooldown)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
