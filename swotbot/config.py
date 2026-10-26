from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWOTBOT_", env_file=".env", extra="ignore")

    data_path: Path = ROOT / "data" / "train.csv"
    model_dir: Path = ROOT / "models"
    max_text_chars: int = 20_000
    max_batch: int = 256


settings = Settings()
