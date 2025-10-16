import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from openai import OpenAI
from tensorzero import patch_openai_client
from tensorzero.util import uuid7

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("tensorzero_model")


@dataclass
class TensorZeroModelConfig:
    config_file: Path


class TensorZeroModel:
    def __init__(self, **kwargs):
        # Remove model_kwargs if present since TensorZero doesn't use it
        kwargs_for_config = {k: v for k,
                             v in kwargs.items() if k != 'model_kwargs'}

        # Determine config file path with priority: env var > kwarg > default bundled config
        config_file_path = (
            os.getenv("TENSORZERO_CONFIG_PATH") or
            kwargs_for_config.get("config_file") or
            str(Path(__file__).parent / "tensorzero.toml")
        )
        # Convert to absolute path
        config_file_path = Path(config_file_path).resolve()

        # Update kwargs with resolved path
        kwargs_for_config["config_file"] = config_file_path

        self.config = TensorZeroModelConfig(**kwargs_for_config)
        self.cost = 0.0
        self.n_calls = 0
        clickhouse_url = os.getenv("TENSORZERO_CLICKHOUSE_URL")
        # TensorZero will override this, but OpenAI client requires something
        client = OpenAI(api_key="dummy-key-for-tensorzero")
        patch_openai_client(
            client, config_file=str(self.config.config_file), clickhouse_url=clickhouse_url, async_setup=False)
        self.client = client
        self.episode_id = uuid7()

    def _query(self, messages: list[dict[str, str]], **kwargs):
        # TensorZero requires model names in format: tensorzero::function_name::XXX
        # Using the swe_agent function defined in the config

        return self.client.chat.completions.create(
            model="tensorzero::function_name::swe_agent", messages=messages, **kwargs, extra_body={"tensorzero::episode_id": str(self.episode_id)}
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        cost = 0
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        content = response.choices[0].message.content or ""
        print("Agent: ", content)
        return {
            # type: ignore
            "content": response.choices[0].message.content or "",
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
