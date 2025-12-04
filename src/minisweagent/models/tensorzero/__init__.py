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

        # Check for HTTP gateway mode
        gateway_url = os.getenv("TENSORZERO_GATEWAY_URL")

        if gateway_url:
            # HTTP gateway mode - config file not required
            self.client = TensorZeroGateway.build_http(gateway_url=gateway_url)
            self.config = None
        else:
            # Embedded gateway mode - existing logic
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
            clickhouse_url = os.getenv("TENSORZERO_CLICKHOUSE_URL")

            # Use native TensorZero client with embedded gateway
            self.client = TensorZeroGateway.build_embedded(
                config_file=str(self.config.config_file), clickhouse_url=clickhouse_url
            )

        self.cost = 0.0
        self.n_calls = 0
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
        # Build structured content blocks and extract text for action parsing
        content_blocks = []
        text_content = ""

        for block in response.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
                text_content += block.text
            elif block.type == "thought":
                thought_block = {"type": "thought", "text": block.text}
                # Include signature if present (required for Anthropic extended thinking round-trip)
                if hasattr(block, "signature") and block.signature:
                    thought_block["signature"] = block.signature
                content_blocks.append(thought_block)
            # Ignore unknown types for now

        print("Agent: ", text_content)

        # Write episode_id to .episode_id file
        Path(".episode_id").write_text(str(self.episode_id))

        return {
            "content": text_content,  # Plain text for action parsing
            "content_blocks": content_blocks,  # Structured for message history
        }

    def get_template_vars(self) -> dict[str, Any]:
        base_vars = {"n_model_calls": self.n_calls, "model_cost": self.cost}
        if self.config:
            return asdict(self.config) | base_vars
        return base_vars
