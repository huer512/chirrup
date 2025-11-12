from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field

from chirrup.core_structure import DEFAULT_SAMPLING_CONFIG, DEFAULT_STOP_TOKENS


# OpenAI API 兼容的数据模型
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色，可以是 'user', 'assistant', 'system'")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="rwkv-latest", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    stream: bool = Field(default=False, description="是否流式返回")

    temperature: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["temperature"],
        ge=0.0,
        le=2.0,
        description="采样温度",
    )
    top_p: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["top_p"],
        ge=0.0,
        le=1.0,
        description="nucleus 采样参数",
    )
    presence_penalty: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["presence_penalty"],
        ge=0,
        le=2.0,
        description="存在惩罚",
    )
    frequency_penalty: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["frequency_penalty"],
        ge=0,
        le=2.0,
        description="频率惩罚",
    )
    penalty_decay: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["penalty_decay"],
        ge=0.0,
        le=1.0,
        description="惩罚衰减",
    )

    max_tokens: int = Field(
        default=DEFAULT_SAMPLING_CONFIG["max_tokens"],
        ge=1,
        description="最大生成 token 数",
    )
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="停止词")

    pad_zero: bool = Field(default=True, description="是否在输入的 prompt token 前添加 0")


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


# Alic 翻译接口


class TranslateRequest(BaseModel):
    source_lang: str = "auto"
    target_lang: str
    text_list: list[str]
    placeholders: list[str] = None


class TranslateResponse(BaseModel):
    translations: list[TranslationResult]
    id: str
    created: int


class TranslationResult(BaseModel):
    text: str
    detected_source_lang: str


# Rollout 接口


class RolloutRequest(BaseModel):
    model: str = Field(default="rwkv-latest", description="模型名称")
    contents: List[str] = Field(..., description="Rollout 列表")
    stream: bool = Field(default=False, description="是否流式返回")

    temperature: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["temperature"],
        ge=0.0,
        le=2.0,
        description="采样温度",
    )
    top_p: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["top_p"],
        ge=0.0,
        le=1.0,
        description="nucleus 采样参数",
    )
    presence_penalty: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["presence_penalty"],
        ge=0,
        le=2.0,
        description="存在惩罚",
    )
    frequency_penalty: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["frequency_penalty"],
        ge=0,
        le=2.0,
        description="频率惩罚",
    )
    penalty_decay: float = Field(
        default=DEFAULT_SAMPLING_CONFIG["penalty_decay"],
        ge=0.0,
        le=1.0,
        description="惩罚衰减",
    )

    max_tokens: int = Field(
        default=DEFAULT_SAMPLING_CONFIG["max_tokens"],
        ge=1,
        description="最大生成 token 数",
    )
    stop_tokens: list[int] = Field(default=DEFAULT_STOP_TOKENS, description="停止 tokens")

    pad_zero: bool = Field(default=True, description="是否在输入的 prompt token 前添加 0")


class RolloutStreamResponse(BaseModel):
    id: str
    object: str = "batch.rollout.chunk"
    created: int
    model: str
    choices: List[RolloutStreamChoice]


class RolloutStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
