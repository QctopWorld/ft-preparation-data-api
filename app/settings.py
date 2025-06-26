# settings.py --------------------------------------------------------------
import os
from dotenv import load_dotenv
from pydantic import  Field
from typing import ClassVar, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    load_dotenv(override=True) # charge les variables d'environnement depuis le fichier .env

    openai_api_key: ClassVar[str] = os.getenv("OPENAI_API_KEY")
    
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
