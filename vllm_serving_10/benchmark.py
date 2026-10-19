import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

prompt = "Explain the difference between data parallelism and tensor parallelism."

start = time.perf_counter()

response = client.chat.completions.create(
    model="Qwen3-14B",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=256,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)

elapsed = time.perf_counter() - start

output_tokens = response.usage.completion_tokens

print(f"Elapsed time: {elapsed:.2f} seconds")
print(f"Output tokens: {output_tokens}")
print(f"Output tokens / total request time: {output_tokens / elapsed:.2f}")