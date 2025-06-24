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
