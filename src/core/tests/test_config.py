import importlib
import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from set.config import Settings, require_llm_settings


class SettingsTests(unittest.TestCase):
    def test_agent_scheduler_uses_small_bounded_default(self) -> None:
        self.assertEqual(Settings().AGENT_SCHEDULER_MAX_WORKERS, 2)

    def test_agent_scheduler_rejects_non_positive_worker_limit(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(AGENT_SCHEDULER_MAX_WORKERS=0)

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
