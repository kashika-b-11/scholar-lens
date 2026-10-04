"""
ScholarLens Configuration Loader.
Parses config.yaml and environment variables into a centralized settings object.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Automatically load .env file from project root or config directory
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
elif (CONFIG_DIR / ".env").exists():
    load_dotenv(dotenv_path=CONFIG_DIR / ".env")
else:
    load_dotenv()


class AppConfig:
    """Manages application and LLM hyperparameter configurations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or (CONFIG_DIR / "config.yaml")
        self._raw_config: Dict[str, Any] = self._load_yaml()
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Loads configuration from YAML file with fallback defaults."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @property
    def app_title(self) -> str:
        return self._raw_config.get("app", {}).get("title", "ScholarLens Copilot")

    @property
    def default_provider(self) -> str:
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider:
            return env_provider.lower()
        return self._raw_config.get("llm", {}).get("default_provider", "gemini")

    @property
    def gemini_model(self) -> str:
        return self._raw_config.get("llm", {}).get("gemini", {}).get("model", "gemini-3.8-flash")

    @property
    def gemini_fallback_model(self) -> str:
        return self._raw_config.get("llm", {}).get("gemini", {}).get("fallback_model", "gemini-flash-latest")


    @property
    def openai_model(self) -> str:
        return self._raw_config.get("llm", {}).get("openai", {}).get("model", "gpt-4o-mini")

    @property
    def temperature(self) -> float:
        return float(self._raw_config.get("llm", {}).get("gemini", {}).get("temperature", 0.2))

    @property
    def max_output_tokens(self) -> int:
        return int(self._raw_config.get("llm", {}).get("gemini", {}).get("max_output_tokens", 3000))

    @property
    def max_char_limit(self) -> int:
        return int(self._raw_config.get("document", {}).get("max_character_limit", 80000))

    @property
    def min_char_limit(self) -> int:
        return int(self._raw_config.get("document", {}).get("min_character_limit", 300))

    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Fetches the corresponding API key from environment variables, reloading .env if needed."""
        # Always reload to pick up newly added keys without requiring server restart
        if ENV_PATH.exists():
            load_dotenv(dotenv_path=ENV_PATH, override=True)
        elif (CONFIG_DIR / ".env").exists():
            load_dotenv(dotenv_path=CONFIG_DIR / ".env", override=True)
        else:
            load_dotenv(override=True)

        prov = (provider or self.default_provider).lower()
        if prov == "gemini":
            key = os.getenv("GEMINI_API_KEY")
            return key.strip() if key else None
        elif prov == "openai":
            key = os.getenv("OPENAI_API_KEY")
            return key.strip() if key else None
        return None



# Global singleton instance for easy import
config = AppConfig()
