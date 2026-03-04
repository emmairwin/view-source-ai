"""Quick test — see raw prompt and model response."""
import sys
import json
sys.stderr = open('/dev/null', 'w')  # suppress warnings

from app import get_llm, build_prompt, SentimentConfig, initialize_example_bank, parse_llm_response

config = SentimentConfig()
bank = initialize_example_bank()
messages = build_prompt('Well you are stupid', config, bank)

print('=== MESSAGES ===')
for m in messages:
    print(f"[{m['role']}]: {m['content'][:400]}")

llm = get_llm()
resp = llm.generate(messages)

print('\n=== RAW RESPONSE ===')
print(repr(resp))
print('\n=== PARSED ===')
print(json.dumps(parse_llm_response(resp, config), indent=2))
