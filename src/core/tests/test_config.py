import importlib
import os
import unittest
from unittest.mock import patch

from set.config import Settings, require_llm_settings


class SettingsTests(unittest.TestCase):
    def test_missing_llm_configuration_is_reported_when_used(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "",
                "DEEPSEEK_BASE_URL": "",
                "LLM_MODEL": "",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing required LLM settings"):
                _require(Settings())


def _require(settings: Settings) -> Settings:
    config_module = importlib.import_module("set.config")
    original = config_module.settings
    try:
        config_module.settings = settings
        return require_llm_settings()
    finally:
        config_module.settings = original


if __name__ == "__main__":
    unittest.main()
