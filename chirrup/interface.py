import asyncio
import torch
import uuid
from typing import Optional, List, Callable, Union, Tuple, Any, Literal, TypedDict

from chirrup.core_structure import Task, DEFAULT_SAMPLING_CONFIG, DEFAULT_STOP_TOKENS


class CachePrefill(TypedDict):
    state: list[torch.Tensor]
    prefilled_tokens: List[int]


TASK_RETURN_TYPE = Union[Tuple[Literal["token"], int, str], Tuple[Literal["cache_prefill"], CachePrefill]]


class AsyncEngineCompletion:
    def __init__(
        self,
        # 输入参数
        prompt_str: str,
        prefill_tokens: List[int],
        state: Union[None | List[torch.Tensor]],
        # 流程控制
        task_queue: asyncio.Queue[Task],
        priority: int = 0,
        # 采样参数
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
        worker_output_queue: Optional[Tuple[str, str, Any]] = asyncio.Queue(),
        task_output_queue: asyncio.Queue[Tuple[str, TASK_RETURN_TYPE]] = asyncio.Queue(),
        cache_prefill: bool = False,
        cache_prefill_padding: int = 0,
    ):
        self.task_id = task_id if task_id else str(uuid.uuid4())

        # 创建任务特定的事件队列
        self.task_event_queue = asyncio.Queue()

        self.task = Task(
            task_id=self.task_id,
            priority=priority,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_decay=penalty_decay,
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
            prompt_str=prompt_str,
            prefill_tokens=prefill_tokens,
            state=state,
            output_queue=worker_output_queue,
            task_event_queue=self.task_event_queue,
            forbidden_tokens=forbidden_tokens,
            cache_prefill=cache_prefill,
            cache_prefill_padding=cache_prefill_padding,
        )
        self.task_output_queue = task_output_queue

        self.task_queue = task_queue

        self.started = False
        self.is_finished = False

        # self.completion_status: Optional[Task] = None

    def start(self):
        self.started = True
        self.task_queue.put_nowait(self.task)

    def __aiter__(self):
        if not self.started:
            self.start()

        return self

    async def __anext__(
        self,
    ) -> TASK_RETURN_TYPE:
        if self.is_finished:
            raise RuntimeError("Already finished")

        while True:
            out = await self.task_output_queue.get()

            if isinstance(out, tuple) and len(out) == 2:
                message_type, payload = out
                if message_type == "token_generated":
                    return ("token", *payload)
                elif message_type == "task_completed":
                    self.is_finished = True
                    self.task = payload
                    raise StopAsyncIteration
                elif message_type == "cache_prefill":
                    return ("cache_prefill", payload)

            else:
                print("Unknown message format:", out)

    def get_full_completion(self) -> asyncio.Task[str]:
        async def fetch_all_tokens() -> str:
            result = []
            async for event in self:
                if event[0] == "token":
                    result.append(event[2])
            return "".join(result)

        return asyncio.create_task(fetch_all_tokens())

    def abort(self):
        self.task_event_queue.put_nowait(("abort", None))
