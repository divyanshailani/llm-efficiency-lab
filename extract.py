import json
with open("/Users/divyanshailani/.gemini/antigravity/brain/6143c27a-921f-4be3-8df8-e06354e8bedd/.system_generated/logs/transcript.jsonl") as f:
    count = 0
    for line in f:
        try: data = json.loads(line)
        except: continue
        if data.get("type") == "USER_INPUT":
            step = data.get('step_index')
            print(f"--- STEP {step} ---")
            print(data.get("content"))
            count += 1
            if count >= 15:
                break
