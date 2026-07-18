import json
import glob

def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            for result in data:
                test = result.get('test')
                n_prompt = result.get('n_prompt')
                tps = result.get('t_s')
                print(f"{filepath.split('/')[-1]} | {test} ({n_prompt}) | {tps:.2f} t/s")
    except Exception as e:
        print(f"Failed to parse {filepath}: {e}")

for file in sorted(glob.glob('/Users/divyanshailani/Desktop/llm experiments/results/mac_mini_m4/kv_scaling/*.json')):
    process_file(file)
