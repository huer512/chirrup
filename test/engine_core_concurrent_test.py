import asyncio
from tqdm import tqdm

from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig


async def test_engine_core():
    """测试 AsyncEngineCore 的基本功能"""

    # 创建模型配置
    model_config = ModelLoadConfig(
        model_path="../models/rwkv7-g0a3-7.2b-20251029-ctx8192.pth",
        # model_path="../models/rwkv7-g1a3-1.5b-20251015-ctx8192",
        vocab_path="../Albatross/reference/rwkv_vocab_v20230424.txt",
        vocab_size=65536,
        head_size=64,
    )

    # 创建引擎核心
    engine_core = AsyncEngineCore()

    try:
        # 测试初始化 Worker
        print("测试初始化 Worker...")
        await engine_core.init(worker_num=1, model_config=model_config, batch_size=33)

        print("测试创建 completion 对象...")

        total = 100
        pbar = tqdm(total=total, unit="Sequence")

        prompts = [f"User: 为什么 {i} 是一个有趣的数字？\n\nAssistant: </think>\n</think>" for i in range(total)]

        async def create_full_completion(prompt):
            result = await engine_core.completion(prompt,cache_prefill=True).get_full_completion()
            # print(prompt)
            pbar.update(1)

        async def logger():
            worker_output_queue = engine_core.worker_pubsub.sub("worker_0")
            prev_mem_alloc = -1
            while True:
                log = await worker_output_queue.get()
                pbar.set_description(
                    f"loop HZ: {(1/log[1]['avg_loop_time']):.3f} | {' '.join([f'{k}:{v}' for k,v in log[1]['task_details'].items()])} | delta mem: {log[1]['max_allocated_memory_GB'] - prev_mem_alloc:.3f} GB"
                )
                prev_mem_alloc = log[1]["max_allocated_memory_GB"]

        log_task = asyncio.create_task(logger())

        print("启动 completion ...")
        results = await asyncio.gather(
            *[
                create_full_completion(
                    prompt,
                )
                for prompt in prompts
            ]
        )

        # print(results)

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 清理资源
        pbar.close()
        print("清理资源...")
        engine_core.shutdown()
        print("✓ 资源清理完成")


if __name__ == "__main__":
    asyncio.run(test_engine_core())
