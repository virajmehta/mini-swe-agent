import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from tensorzero import TensorZeroGateway
from tensorzero.util import uuid7

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("tensorzero_model")


@dataclass
class TensorZeroModelConfig:
    config_file: Path


class TensorZeroModel:
    def __init__(self, **kwargs):
        # Extract tags if provided
        self.tags = kwargs.pop("tags", {})

        # Remove model_kwargs and model_name if present since TensorZero doesn't use them
        kwargs_for_config = {k: v for k, v in kwargs.items()
                             if k not in ("model_kwargs", "model_name")}

        # Determine config file path with priority: env var > kwarg > default bundled config
        config_file_path = (
            os.getenv("TENSORZERO_CONFIG_PATH")
            or kwargs_for_config.get("config_file")
            or str(Path(__file__).parent / "config" / "tensorzero.toml")
        )
        # Convert to absolute path
        config_file_path = Path(config_file_path).resolve()

        # Update kwargs with resolved path
        kwargs_for_config["config_file"] = config_file_path

        self.config = TensorZeroModelConfig(**kwargs_for_config)
        self.cost = 0.0
        self.n_calls = 0
        clickhouse_url = os.getenv("TENSORZERO_CLICKHOUSE_URL")

        # Use native TensorZero client with embedded gateway
        self.client = TensorZeroGateway.build_embedded(
            config_file=str(self.config.config_file), clickhouse_url=clickhouse_url
        )
        self.episode_id = uuid7()

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        # Use native TensorZero inference API
        inference_kwargs = {"tags": self.tags} if self.tags else {}
        inference_kwargs.update(kwargs)
        response = self.client.inference(
            function_name="swe_agent", input={"messages": messages}, episode_id=str(self.episode_id), **inference_kwargs
        )

        cost = 0
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        # Extract content from TensorZero response
        content = response.content[0].text if response.content else ""
        print("Agent: ", content)

        return {
            "content": content,
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
