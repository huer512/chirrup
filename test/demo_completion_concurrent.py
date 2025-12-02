import asyncio
import time
from openai import AsyncOpenAI

async def send_request(client: AsyncOpenAI, prompt, request_id):
    try:
        response = await client.completions.create(
            model="rwkv-latest",
            prompt=prompt,
            max_tokens=50,
        )
        content = response.choices[0].text
        print(f"Request {request_id}: {content}")
        return content
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-test-key"
    )

    tasks = []
    for i in range(10):
        prompt = f"Number {i}:"
        task = send_request(client, prompt, i)
        tasks.append(task)

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    successful_requests = sum(1 for result in results if result is not None)
    print(f"\nCompleted {successful_requests}/{len(tasks)} requests successfully")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())

