import asyncio
import time
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-test-key"
    )

    # 测试非流式补全
    print("测试非流式补全:")
    start = time.perf_counter()
    response = await client.completions.create(
        model="rwkv-latest",
        prompt="Once upon a time",
        max_tokens=100,
        temperature=1.0,
    )
    print(response.choices[0].text)
    print(f"耗时: {time.perf_counter() - start:.2f}s")
    print(f"Tokens: {response.usage.completion_tokens}")
    
    print("\n" + "="*60 + "\n")
    
    # 测试流式补全
    print("测试流式补全:")
    stream = await client.completions.create(
        model="rwkv-latest",
        prompt="The quick brown fox",
        max_tokens=100,
        temperature=1.0,
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices[0].text:
            print(chunk.choices[0].text, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())

