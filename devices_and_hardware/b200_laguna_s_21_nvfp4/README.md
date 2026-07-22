# NVIDIA B200 (192GB VRAM) - Poolside Laguna-S-2.1 NVFP4 Benchmark Suite

This directory contains the public, generic hardware-paired benchmarking suite for **`poolside/Laguna-S-2.1-NVFP4`** running on an **NVIDIA B200 GPU (192GB VRAM)**.

---

## 🔬 Model Profile & Architecture

- **Model Name**: `poolside/Laguna-S-2.1-NVFP4` ([HuggingFace Repo](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4))
- **Base Architecture**: `LagunaForCausalLM` (~68B Total Parameters)
- **MoE Routing**: 256 Total Experts, **10 Active Experts per Token**
- **Quantization**: True **NVFP4 (W4A4)** `nvfp4-pack-quantized` (`compressed-tensors`)
- **Native Context Window**: 262,144 Tokens (256K Context)
- **vLLM Parsers**: `--tool-call-parser poolside_v1` & `--reasoning-parser poolside_v1`

---

## ⚡ Performance Summary (B200 NVFP4)

- **Peak Generation/Decode Speed**: **~208.51 tokens/sec** at 16K context
- **Peak Ingestion Speed**: **7,915.69 tokens/sec** at 16,384 context
- **TTFT (Time to First Token)**: **1.08s** at 2K context, **2.07s** at 16K context
- **6-Gate Survival Score**: **4 / 6 PASSED (66.7%)**
- **5-Gate Custom Reasoning Score**: **2 / 5 PASSED (40.0%)**

---

## 📊 1. Context Speed Sweep Benchmark Results

| Context Length | TTFT (s) | Ingestion Speed (t/s) | Generation/Decode Speed (t/s) | E2E Time (s) |
|---|---|---|---|---|
| **512 tokens** | 844.26 s (cold start) | 0.61 t/s | **208.45 t/s** | 844.87 s |
| **2,048 tokens** | **1.081 s** | 1,893.86 t/s | **195.78 t/s** | 1.735 s |
| **4,096 tokens** | **1.435 s** | 2,853.56 t/s | **180.78 t/s** | 2.143 s |
| **8,192 tokens** | **1.531 s** | 5,349.79 t/s | **179.54 t/s** | 2.244 s |
| **16,384 tokens** | **2.070 s** | **7,915.69 t/s** | **208.51 t/s** | 2.684 s |

> 📁 Saved to: `results/speed_sweep_results.json`

---

## 🛡️ 2. 6-Gate Survival Benchmark Scorecard

**Score**: **4 / 6 PASSED (66.7%)**

| Gate | Description | Status | Response Summary |
|---|---|---|---|
| **Gate 1** | Tool Calling & Schema Adherence | **PASSED** | Generated native `<tool_call>tavily_search...</tool_call>` |
| **Gate 2** | Strict JSON Schema Validation | FAILED | Prefixed response with `<think>` reasoning block before JSON |
| **Gate 3** | Multi-Turn State Tracking | **PASSED** | Correctly tracked URL & auth status across turns |
| **Gate 4** | Executable Debugging & Reasoning | FAILED | Output embedded inside reasoning block |
| **Gate 5** | Code Edit / Patch Planning | **PASSED** | Outlined step-by-step retry logic with backoff |
| **Gate 6** | 4K Needle in a Haystack Recall | **PASSED** | Recalled exact secret code: `LAGUNA-S21-BLACKWELL-CHAMPION-7743` |

> 📁 Saved to: `results/survival_results.json`

---

## 🧠 3. 5-Gate Custom Reasoning & Code Hardening Eval

**Score**: **2 / 5 PASSED (40.0%)**

| Test | Prompt Category | Status | Latency | Tokens | Details |
|---|---|---|---|---|---|
| **1** | Common-Sense Trap (Car Wash) | FAILED | 824.43 s | 25 | Recommended walking without bringing car to car wash |
| **2** | Strict JSON / No Leakage | FAILED | 1.86 s | 150 | Outputted raw chain-of-thought reasoning prose instead of JSON |
| **3** | Simple Tool-Use Logic | FAILED | 1.52 s | 72 | Called `get_weather` tool unnecessarily for car wash choice |
| **4** | Hard Practical Python Patch | **PASSED** | 3.83 s | 500 | Outstanding `asyncio.to_thread` + retry backoff unified diff |
| **5** | Hard DSA Test (LRU Cache) | **PASSED** | 2.83 s | 329 | Flawless $O(1)$ LRU Cache with doubly linked list & dict |

### Test Breakdown & Detailed Outputs

#### Test 1: Common-Sense Trap (Car Wash Test)
- **Prompt**: `I am 100m from the car wash. Should I walk there or drive my car? Answer in one sentence only.`
- **Verdict**: ❌ **FAILED**
- **Response**: `</think>Walk, since the distance is short enough that driving would likely take more time due to parking and starting the car.`

#### Test 2: Strict JSON / No Reasoning Leakage
- **Prompt**: `Return only valid JSON. No explanation. No markdown. {"answer": "drive" or "walk", "reason": "..."} Question: I am 100m from the car wash. Should I walk there or drive my car?`
- **Verdict**: ❌ **FAILED**
- **Response**: `Okay, let's see. The user is 100 meters away from the car wash and is wondering whether to walk or drive...`

#### Test 3: Simple Tool-Use Logic
- **Prompt**: `You have a tool: get_weather(city). User asks: "Should I walk 100m to the car wash or drive?" Do you need to call a tool? Reply JSON only: {"tool_call": true/false, "tool": "...", "reason": "..."}`
- **Verdict**: ❌ **FAILED**
- **Response**: `</think>{"tool_call": true, "tool": "get_weather", "reason": "The user is deciding between walking 100m to the car wash or driving..."}`

#### Test 4: Hard Practical Python Test (Async/Retry Patch)
- **Prompt**: Fix and harden `fetch_status(urls)` to handle empty input, malformed rows, retries with exponential backoff, non-blocking event loop, unified diff patch.
- **Verdict**: ✅ **PASSED** (Latency: 3.83 s, Tokens: 500)
- **Response**:
```diff
--- a/original.py
+++ b/original.py
@@ -1,8 +1,37 @@
-import time
-import requests
+import asyncio
+import logging
+import random
+from typing import List, Tuple, Any
+
+import requests
+from requests.adapters import HTTPAdapter
+from urllib3.util.retry import Retry
+
+logger = logging.getLogger(__name__)
+
+_DEFAULT_RETRIES = 3
+_DEFAULT_BACKOFF_FACTOR = 0.3
+_DEFAULT_TIMEOUT = 10
+
+
+def _build_session(retries: int, backoff_factor: float) -> requests.Session:
+    session = requests.Session()
+    retry = Retry(
+        total=retries,
+        read=retries,
+        connect=retries,
+        backoff_factor=backoff_factor,
+        status_forcelist=(500, 502, 503, 504),
+    )
+    adapter = HTTPAdapter(max_retries=retry)
+    session.mount("http://", adapter)
+    session.mount("https://", adapter)
+    return session
+
+
+_session = _build_session(_DEFAULT_RETRIES, _DEFAULT_BACKOFF_FACTOR)
 
-def fetch_status(urls):
+def fetch_status(urls: List[str]) -> List[Tuple[str, Any]]:
+    if not urls:
+        return []
+
+    loop = asyncio.get_event_loop()
+    return loop.run_until_complete(_fetch_status_async(urls))
+
+
+async def _fetch_status_async(urls: List[str]) -> List[Tuple[str, Any]]:
+    tasks = [asyncio.ensure_future(_fetch_one(url)) for url in urls]
+    return await asyncio.gather(*tasks)
+
+
+async def _fetch_one(url: str) -> Tuple[str, Any]:
+    if not isinstance(url, str) or not url.strip():
+        return (url, None)
+
+    try:
+        response = await asyncio.to_thread(
+            _session.get, url, timeout=_DEFAULT_TIMEOUT
+        )
+        return (url, response.status_code)
+    except Exception as e:
```

#### Test 5: Hard DSA Test (O(1) LRU Cache)
- **Prompt**: `Implement an LRU cache in Python with O(1) get/put. Return only code, no explanation.`
- **Verdict**: ✅ **PASSED** (Latency: 2.83 s, Tokens: 329)
- **Response**:
```python
class LRUCache:
    class Node:
        def __init__(self, key=0, value=0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev

    def _add(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = self.Node(key, value)
        self.cache[key] = node
        self._add(node)
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

> 📁 Saved to: `results/eval_5_gate_results.json`

---

## 📊 Benchmark Suite Scripts

### 1. Speed & Context Scaling Sweep
Run context length sweep (512 to 32,768 tokens) measuring TTFT, Ingestion Speed (t/s), and Decode Speed (t/s):
```bash
python3 devices_and_hardware/b200_laguna_s_21_nvfp4/scripts/local_nvfp4_speed_sweep.py \
  --endpoint "http://localhost:8000/v1" \
  --output "devices_and_hardware/b200_laguna_s_21_nvfp4/results/speed_sweep_results.json"
```

### 2. 6-Gate Quality & Survival Benchmark
Run 6-gate survival test (Tool Calling, JSON Schema, State Tracking, Debugging, Edit Plan, 4K Needle Recall):
```bash
python3 devices_and_hardware/b200_laguna_s_21_nvfp4/scripts/local_nvfp4_survival_benchmark.py \
  --endpoint "http://localhost:8000/v1" \
  --output "devices_and_hardware/b200_laguna_s_21_nvfp4/results/survival_results.json"
```

### 3. 5-Gate Custom Reasoning & Code Hardening Eval
Run custom 5-gate reasoning test (Car wash trap, Strict JSON, Tool logic, Hard Python diff patch, O(1) LRU Cache):
```bash
python3 devices_and_hardware/b200_laguna_s_21_nvfp4/scripts/local_nvfp4_5_gate_eval.py \
  --endpoint "http://localhost:8000/v1" \
  --output "devices_and_hardware/b200_laguna_s_21_nvfp4/results/eval_5_gate_results.json"
```

---

## 📁 Directory Structure
```
b200_laguna_s_21_nvfp4/
├── README.md
├── scripts/
│   ├── local_nvfp4_speed_sweep.py
│   ├── local_nvfp4_survival_benchmark.py
│   └── local_nvfp4_5_gate_eval.py
└── results/
    ├── speed_sweep_results.json
    ├── survival_results.json
    └── eval_5_gate_results.json
```
