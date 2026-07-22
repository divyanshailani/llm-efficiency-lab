# AGENTS.md - Security & Repository Rules for AI Agents

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
