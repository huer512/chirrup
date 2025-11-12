import asyncio
import queue
import threading
import time
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
import uuid
import torch

from chirrup.core_structure import (
    Task,
    ModelLoadConfig,
    DEFAULT_SAMPLING_CONFIG,
    DEFAULT_STOP_TOKENS,
)
from chirrup.interface import AsyncEngineCompletion
from chirrup.worker import Worker, TRIE_TOKENIZER


class SimplePubSub:
    """简单的发布订阅系统，用于线程间通信"""

    def __init__(self):
        self.subscribers: Dict[str, Tuple[asyncio.Queue, int]] = {}
        self.publishers: queue.Queue[Union[Tuple[str, int, str], Task]] = queue.Queue()
        self.stop_flag: bool = False

    def sub(self, task_id: str, temporary=False) -> asyncio.Queue:
        """订阅特定任务ID的消息"""
        if task_id in self.subscribers:
            if not temporary:
                self.subscribers[task_id][1] += 1
        else:
            self.subscribers[task_id] = [asyncio.Queue(), 1]

        return self.subscribers[task_id][0]

    def thread_safe_loop(self, event_loop: asyncio.AbstractEventLoop):
        """线程安全的事件循环，用于在线程间传递消息"""

        def safe_put(target: asyncio.Queue, item):
            try:
                target.put_nowait(item)
            except RuntimeError:
                # 如果队列已关闭或事件循环已关闭，则忽略
                pass

        while True:
            if self.stop_flag:
                break

            tasks = []
            while True:
                try:
                    tasks.append(self.publishers.get_nowait())
                except queue.Empty:
                    break

            for task in tasks:
                if task is None:
                    continue

                # 统一处理所有消息格式为 (target_id, message_type, payload)
                if isinstance(task, tuple) and len(task) == 3:
                    target_id, message_type, payload = task
                    if target_id in self.subscribers:
                        try:
                            event_loop.call_soon_threadsafe(
                                safe_put,
                                self.subscribers[target_id][0],
                                (message_type, payload),
                            )
                            # 如果是任务完成消息，取消订阅
                            if message_type == "task_completed":
                                del self.subscribers[target_id]
                        except RuntimeError:
                            # 事件循环已关闭，忽略任务
                            pass
                    else:
                        del payload
                else:
                    print("Unknown message format:", task)

            time.sleep(0.01)

    def unsub(self, task_id: str) -> bool:
        """取消订阅特定任务ID的消息，释放并移除对应的订阅者

        Args:
            task_id: 要取消订阅的任务ID

        Returns:
            bool: 如果成功取消订阅返回True，否则返回False
        """
        if task_id in self.subscribers:
            self.subscribers[task_id][1] -= 1
            if self.subscribers[task_id][1] <= 0:
                del self.subscribers[task_id]
                return True
        return False

    def stop(self):
        """停止 pub/sub 系统"""
        self.stop_flag = True
        self.publishers.put_nowait(None)


class AsyncEngineCore:
    """
    核心引擎类，负责处理核心功能，包括：
    - Worker 管理
    - Task 管理
    - 实现 pub/sub 功能
    """

    def __init__(self):
        self.workers: List[Worker] = []
        self.worker_threads: List[threading.Thread] = []
        self.task_queue: queue.Queue[Task] = queue.Queue()
        self.event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

        self.worker_id_set = set()

        # pub/sub 系统
        self.pubsub = SimplePubSub()
        self.worker_pubsub = SimplePubSub()

        # 线程管理
        self.pubsub_thread: Optional[threading.Thread] = None
        self.worker_pubsub_thread: Optional[threading.Thread] = None

        # 事件循环引用
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

        # 初始化状态
        self.is_initialized = False
        self.is_shutdown = False

        # 初始化分词器
        self.tokenizer: TRIE_TOKENIZER = None

    def init(self, worker_num: int, model_config: ModelLoadConfig, batch_size: int = 32) -> asyncio.Task:
        """
        初始化 Worker，返回一个异步任务，当全部 worker 都加载成功后完成

        Args:
            worker_num: Worker 数量
            model_config: 模型配置
            batch_size: 批处理大小

        Returns:
            asyncio.Task: 当所有 worker 加载完成后完成的异步任务
        """
        if self.is_initialized:
            raise RuntimeError("Workers already initialized")

        if self.is_shutdown:
            raise RuntimeError("Engine has been shutdown")

        # 获取当前事件循环
        try:
            self.event_loop = asyncio.get_running_loop()
        except RuntimeError:
            # 如果没有运行的事件循环，创建一个新的
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)

        # 启动 pub/sub 线程
        self.pubsub_thread = threading.Thread(target=self.pubsub.thread_safe_loop, args=(self.event_loop,))
        self.pubsub_thread.start()

        self.worker_pubsub_thread = threading.Thread(
            target=self.worker_pubsub.thread_safe_loop, args=(self.event_loop,)
        )
        self.worker_pubsub_thread.start()

        self.is_initialized = True

        self.tokenizer = TRIE_TOKENIZER(model_config.vocab_path)

        # 创建异步任务来等待所有 worker 加载完成
        async def wait_for_workers_loaded():
            """等待所有 worker 加载完成的异步任务"""
            # 创建一个队列来接收 worker 加载完成的消息
            loaded_queue = asyncio.Queue()

            self.worker_id_set = {f"worker_{i}" for i in range(worker_num)}

            # 订阅 worker 加载事件的异步任务
            async def worker_event_handler():
                while True:
                    await asyncio.sleep(1)
                    for worker_id in self.worker_id_set:
                        try:
                            message_type, payload = self.worker_pubsub.sub(worker_id, temporary=True).get_nowait()
                            if message_type == "worker_loaded":
                                await loaded_queue.put((worker_id, payload))
                        except asyncio.QueueEmpty:
                            continue

            # 启动事件处理任务
            event_handler_task = asyncio.create_task(worker_event_handler())

            # 创建并启动 Worker 线程

            for k, worker_id in enumerate(self.worker_id_set):
                gpu_id = [k]  # 假设每个 Worker 使用一个 GPU

                worker = Worker(
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    model_config=model_config,
                    task_queue=self.task_queue,
                    master_event_queue=self.event_queue,
                    worker_event_queue=self.worker_pubsub.publishers,
                    batch_size=batch_size,
                )

                self.workers.append(worker)

                # 在独立线程中启动 Worker
                worker_thread = threading.Thread(target=worker.start, daemon=True, name=f"chirrup:{worker_id}")
                worker_thread.start()
                self.worker_threads.append(worker_thread)

            # 等待所有 worker 加载完成
            loaded_workers = set()
            timeout = 300  # 5分钟超时

            try:
                while len(loaded_workers) < worker_num and timeout > 0:
                    try:
                        worker_id, event = await asyncio.wait_for(loaded_queue.get(), timeout=1.0)
                        if event.get("status") == "success":
                            loaded_workers.add(worker_id)
                        else:
                            print(f"Worker {worker_id} 加载失败: {event}")
                            raise RuntimeError(f"Worker {worker_id} failed to load")
                    except asyncio.TimeoutError:
                        timeout -= 1

                if len(loaded_workers) < worker_num:
                    failed_workers = self.worker_id_set - loaded_workers
                    raise RuntimeError(f"以下 worker 加载超时: {failed_workers}")

                print(f"所有 {worker_num} 个 worker 加载完成")
            finally:
                # 等待事件处理任务完成
                event_handler_task.cancel()

            for worker_id in self.worker_id_set:
                self.worker_pubsub.unsub(worker_id)

            # 创建并返回异步任务

        return asyncio.create_task(wait_for_workers_loaded())

    def completion(
        self,
        prompt_str: str,
        prefill_tokens: Optional[List[int]] = None,
        state: Optional[Union[None, List[torch.Tensor]]] = None,
        priority: int = 0,
        temperature: float = DEFAULT_SAMPLING_CONFIG["temperature"],
        top_p: float = DEFAULT_SAMPLING_CONFIG["top_p"],
        top_k: int = DEFAULT_SAMPLING_CONFIG["top_k"],
        presence_penalty: float = DEFAULT_SAMPLING_CONFIG["presence_penalty"],
        frequency_penalty: float = DEFAULT_SAMPLING_CONFIG["frequency_penalty"],
        penalty_decay: float = DEFAULT_SAMPLING_CONFIG["penalty_decay"],
        stop_tokens: Optional[List[int]] = DEFAULT_STOP_TOKENS,
        forbidden_tokens: Optional[List[int]] = [],
        max_tokens: Optional[int] = DEFAULT_SAMPLING_CONFIG["max_tokens"],
        task_id: Optional[str] = None,
        cache_prefill: bool = False,
        cache_prefill_padding: int = 0,
    ) -> AsyncEngineCompletion:
        """
        创建一个 AsyncEngineCompletion 对象，并输入相应配置信息

        Args:
            prompt_str: 提示字符串
            prefill_tokens: 输入token列表
            state: 模型状态
            priority: 任务优先级
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            presence_penalty: 存在惩罚
            frequency_penalty: 频率惩罚
            penalty_decay: 惩罚衰减
            stop_tokens: 停止token列表
            forbidden_tokens: 禁用token列表
            max_tokens: 最大生成token数
            task_id: 任务ID，如果不提供则自动生成

        Returns:
            AsyncEngineCompletion 对象
        """
        assert not (
            state is not None and prefill_tokens is None
        ), "prefill_tokens cannot be None when state is not None"

        if not self.is_initialized:
            raise RuntimeError("Engine not initialized")

        if self.is_shutdown:
            raise RuntimeError("Engine has been shutdown")

        if task_id is None:
            task_id = str(uuid.uuid4())

        if not prefill_tokens:
            prefill_tokens = self.tokenizer.encode(prompt_str)

        # 创建 AsyncEngineCompletion 对象
        completion = AsyncEngineCompletion(
            prompt_str=prompt_str,
            prefill_tokens=prefill_tokens,
            state=state,
            task_queue=self.task_queue,
            priority=priority,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_decay=penalty_decay,
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
            task_id=task_id,
            worker_output_queue=self.pubsub.publishers,
            task_output_queue=self.pubsub.sub(task_id),
            forbidden_tokens=forbidden_tokens,
            cache_prefill=cache_prefill,
            cache_prefill_padding=cache_prefill_padding,
        )

        return completion

    def shutdown(self) -> None:
        """
        关闭引擎，清理资源
        """
        if self.is_shutdown:
            return

        self.is_shutdown = True

        # 发送关闭信号给所有 Worker
        shutdown_event = {"type": "shutdown"}
        try:
            self.event_queue.put_nowait(shutdown_event)
        except Exception as e:
            print(f"Failed to send shutdown signal: {e}")

        # 停止 pub/sub 系统
        self.pubsub.stop()
        self.worker_pubsub.stop()

        # 等待线程结束
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        if self.pubsub_thread and self.pubsub_thread.is_alive():
            self.pubsub_thread.join(timeout=5)

        if self.worker_pubsub_thread and self.worker_pubsub_thread.is_alive():
            self.worker_pubsub_thread.join(timeout=5)

    def __del__(self):
        """析构函数，确保资源被清理"""
        try:
            self.shutdown()
        except Exception:
            pass  # 析构函数中忽略异常
