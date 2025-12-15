import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

class Config(BaseSettings):
    cohere_api_key: Optional[str] = None
    # chroma_db_dir: str = "./chroma_db"
    chroma_db_dir: str = "./chroma_data"  # ✅ Ruta persistente


    class Config:
        env_file = None

config = Config()