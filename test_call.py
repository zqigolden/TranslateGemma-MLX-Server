from openai import OpenAI
import sys

client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="dummy")

text = """Alice was beginning to get very tired of sitting by her sister..."""
if len(sys.argv) > 1:
    text = sys.argv[1]

resp = client.chat.completions.create(
    model="translategemma-mlx",
    messages=[{"role": "user", "content": text}],
    temperature=0.0,
    max_tokens=512,  # Optional; omit to use the server default RESERVED_OUTPUT_TOKENS
)

print(resp.choices[0].message.content)
