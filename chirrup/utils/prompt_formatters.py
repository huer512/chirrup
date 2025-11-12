from typing import List, Tuple, Dict, Any, Union

from chirrup.web_service.api_model import ChatMessage

import re


def clean_openai_message(messages: List[Union[Dict[str, str], ChatMessage]]) -> str:
    """
    将OpenAI API的消息格式化为不思考模式

    Args:
        messages: OpenAI格式的消息列表，每个消息包含role和content键

    Returns:
        格式化后的字符串，不包含思考过程
    """
    formatted_parts = []

    for msg in messages:
        if isinstance(msg, ChatMessage):
            msg = msg.model_dump()
        role = msg["role"]
        content = msg["content"].strip() if msg["content"] else ""
        content = re.sub(r"\n+", "\n", content) if content else ""

        if role == "user":
            formatted_parts.append(f"User: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")
        elif role == "system":
            formatted_parts.append(f"System: {content}")
        else:
            formatted_parts.append(f"{role}: {content}")

    return "\n\n".join(formatted_parts)


def format_openai_message_no_thinking(messages: List[Dict[str, str]]) -> str:
    return clean_openai_message(messages) + "\n\nAssistant:"


def format_openai_message_with_thinking(messages: List[Dict[str, str]]) -> str:
    return clean_openai_message(messages) + "\n\nAssistant:<think>"


def format_openai_message_quick_thinking(messages: List[Dict[str, str]]) -> str:
    return clean_openai_message(messages) + "\n\nAssistant:<think>\n</think>"
