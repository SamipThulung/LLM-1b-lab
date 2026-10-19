from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen3-14B",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        },
        {
            "role": "user",
            "content": "Explain how KV cache works in transformers.",
        },
    ],
    temperature=0.7,
    max_tokens=256,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)

print(response.choices[0].message.content)

print("\nUsage:")
print(response.usage)