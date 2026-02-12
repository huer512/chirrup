import asyncio
import time
import random
from openai import AsyncOpenAI

def generate_random_prompt(request_id):
    """生成一个更长的随机prompt"""
    topics = [
        "人工智能的发展历史",
        "量子计算的原理和应用",
        "区块链技术的未来",
        "机器学习在医疗领域的应用",
        "气候变化对生态系统的影响",
        "太空探索的最新进展",
        "可再生能源技术的发展",
        "生物技术的突破性发现",
        "虚拟现实技术的应用场景",
        "网络安全的重要性"
    ]
    
    questions = [
        "请详细解释一下",
        "你能告诉我关于",
        "我想了解",
        "请描述一下",
        "能否分析一下",
        "请讨论一下",
        "我想知道",
        "请介绍一下",
        "能否解释一下",
        "请阐述一下"
    ]
    
    adjectives = [
        "深入的", "全面的", "详细的", "系统的", "专业的",
        "通俗易懂的", "简洁明了的", "全面的", "深入的", "全面的"
    ]
    
    topic = random.choice(topics)
    question = random.choice(questions)
    adjective = random.choice(adjectives)
    
    prompt = f"请求编号 {request_id}: {question}{topic}，请提供{adjective}信息和分析。"
    return prompt

async def send_request(client: AsyncOpenAI, prompt, request_id):
    try:
        response = await client.completions.create(
            model="rwkv-latest",
            prompt=prompt,
            max_tokens=1024,
        )
        content = response.choices[0].text
        print(f"Request {request_id}: {content}")
        return content
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8001/v1",
        api_key="sk-test-key"
    )

    tasks = []
    for i in range(1,300):
        prompt = generate_random_prompt(i)
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

