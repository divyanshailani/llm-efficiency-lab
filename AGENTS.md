# Global Agent Operating Instructions (Claude, Codex, Antigravity, Local LLMs)

> **MANDATORY**: These rules govern all behavior. Apply them automatically without asking. Any Local LLM agents (Ollama, LM Studio) interacting via MCP must adhere to the same schema.

## 🛑 CRITICAL SECURITY & PRIVACY DIRECTIVES

### 1. ZERO LIVE ENDPOINT URL COMMITS (STRICT SECURITY RULE)
- **NEVER** commit live cloud or Modal deployment URLs (`*.modal.run`, custom deployment domain URLs) to Git.
- Live endpoint URLs exposed in public commits allow web crawlers to drain paid Modal credits.
- **ALL** public scripts and result JSON files MUST use generic placeholders (`http://localhost:8000/v1` or `https://your-endpoint.modal.run/v1`).

### 2. STRICT DIRECTORY ISOLATION
- **`modal_deployments/`**: Contains personal Modal container setup & weight downloaders. **MUST STAY IN `.gitignore`**.
- **`benchmarks/`**: Contains personal test runners & active local execution scripts. **MUST STAY IN `.gitignore`**.
- **`devices_and_hardware/<device>_<model>/`**: Contains public, hardware-paired benchmark documentation & generic tester scripts. **MUST NEVER CONTAIN LIVE API KEYS OR LIVE ENDPOINT URLS**.

### 3. PRE-COMMIT AUDIT PROTOCOL
- Before running `git commit`, the agent **MUST** perform a sweep (`grep -r "modal.run" .`) to verify that zero live Modal deployment URLs are staged.

---

## 🛠️ CORE OPERATING PROTOCOLS

1. **Session Bootstrapping (Omega Memory)**
   - Immediately execute the `omega_protocol` MCP tool at the start of every session to retrieve current dynamic rules, context, and user state.

2. **Code & Architecture (Ponytail Mode)**
   - For any coding or architectural work, you must strictly follow **Ponytail Mode**: Write the absolute minimal, simplest, and laziest solution that actually works. Zero over-engineering. Prefer standard libraries over dependencies. Do not speculate on future needs (YAGNI).

3. **Knowledge Graph Sync (Graph Memory)**
   - After completing *any* code or architectural change, you must automatically log the update to the local project's Graph Memory (using `graph-memory` tools) to keep the codebase blueprint synchronized.
   - **MANDATORY SCHEMA**: When adding nodes or edges, inject the following metadata into your `[attributes_json]`:
     - **Provenance**: `{"created_by": "<Your Agent Name>", "source": "<AST/Human/Web/Tool>"}`
     - **Trust Protocol**: `{"confidence": <0.0-1.0>, "verification_source": "<tool/command>"}`
   - **Execution Workflows**: Do not log every terminal command. Instead, log *macro-level workflows* and completed tasks as `Episode` nodes.

4. **Communication Style (Caveman vs. Standard)**
   - **Normal Chats:** Default to **Caveman Mode** (extreme brevity, no fluff, minimal tokens). 
   - **Technical/Project Discussions:** Disable Caveman mode. Explain complex architectures, bugs, and implementations thoroughly and clearly.

5. **Token Efficiency (Headroom)**
   - If you need to read or scan any large file, repository, or log dump, you must use the **Headroom** tool to compress and extract the context efficiently before processing it.
