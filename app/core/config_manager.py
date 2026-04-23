from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / 'config'
DEFAULT_CONFIG_PATH = CONFIG_DIR / 'settings.yaml'


@dataclass
class AppConfig:
    base_url: str = 'http://127.0.0.1:8000/v1'
    api_key: str = ''
    model_name: str = ''
    system_prompt: str = (
        'You are a careful coding assistant. '
        'Analyze the current code and issue, then produce a corrected solution.'
    )
    temperature: float = 0.2
    max_tokens: int = 4096
    stream: bool = True
    timeout_seconds: int = 120


class ConfigManager:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AppConfig:
        if not self.config_path.exists():
            config = AppConfig()
            self.save(config)
            return config

        with self.config_path.open('r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        defaults = asdict(AppConfig())
        defaults.update(raw)
        return AppConfig(**defaults)

    def save(self, config: AppConfig) -> None:
        with self.config_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(asdict(config), f, allow_unicode=True, sort_keys=False)
