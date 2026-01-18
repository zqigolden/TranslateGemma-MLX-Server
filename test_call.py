from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="lm-studio")

text = """Alice was beginning to get very tired of sitting by her sister..."""

resp = client.chat.completions.create(
    model="translategemma-mlx",
    messages=[{"role": "user", "content": text}],
    temperature=0.0,
    max_tokens=512,  # Optional; omit to use the server default RESERVED_OUTPUT_TOKENS
)

print(resp.choices[0].message.content)
