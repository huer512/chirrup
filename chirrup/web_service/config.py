from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings, cli_parse_args=True, cli_use_class_docs_for_groups=True):
    model_path: str = Field(default="../models/rwkv7-g0a3-7.2b-20251029-ctx8192.pth", description="模型文件路径")
    # model_path: str = Field(default="../models/rwkv7-g1a3-1.5b-20251015-ctx8192.pth", description="模型文件路径")
    vocab_path: str = Field(default="./Albatross/reference/rwkv_vocab_v20230424.txt", description="词汇表文件路径")
    vocab_size: int = Field(default=65536, description="词汇表大小")
    head_size: int = Field(default=64, description="头大小")

    worker_num: int = Field(default=1, ge=1, description="Worker 数量")
    batch_size: int = Field(default=24, ge=1, description="批处理大小")
    state_cache_size: int = Field(default=50, ge=0, description="状态缓存大小")

    host: str = Field(default="127.0.0.1", description="服务器主机地址")
    port: int = Field(default=8000, ge=1, le=65535, description="服务器端口")


# 全局配置实例
CONFIG: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global CONFIG

    if CONFIG is None:
        CONFIG = Config()
    return CONFIG
