# settings.py --------------------------------------------------------------
from pydantic import  Field
from typing import ClassVar, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: ClassVar[str] = "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"
    prefixes_to_remove: ClassVar[List[str]] = [
        "ft:gpt-4.1-mini-2025-04-14:quebectop-inc:",
        "ft:gpt-4o-mini-2024-07-18:quebectop-inc:",
        "ft:gpt-4.1-mini-2025-04-14:quebectop-2007-inc:"
    ]
    weight_up: ClassVar[float] = 3.0
    weight_down: ClassVar[float] = 0.5
    max_workers: ClassVar[int] = 20

    class Config:
        env_file = ".env"      # charge automatiquement OPENAI_API_KEY
        env_file_encoding = "utf-8"
