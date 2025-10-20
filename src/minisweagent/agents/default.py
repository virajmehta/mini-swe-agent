"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'. "
        "If a command needs more time, add '# timeout: <seconds>' on the first line (max {{max_timeout}} seconds)."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
    max_timeout: int = 300


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}

    def get_template_arguments(self, **kwargs) -> dict:
        """Get all template arguments by merging config, env, model, and extra vars."""
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        all_vars = {**template_vars, **self.extra_template_vars, **kwargs}

        # Convert Path objects to strings for JSON serialization
        return {k: str(v) if isinstance(v, Path) else v for k, v in all_vars.items()}

    def add_message(self, role: str, content: str | list[dict], **kwargs):
        """Add a message with either string content or template content blocks."""
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []

        # TensorZero handles system prompts via variant config, not as a message
        # So we only add the user message with the task/instance template
        instance_args = self.get_template_arguments()
        self.add_message("user", [{"type": "template", "name": "instance", "arguments": instance_args}])

        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))

        # Add observation message with template block
        observation_args = {"output": output}
        self.add_message("user", [{"type": "template", "name": "action_observation", "arguments": observation_args}])
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            action_text = actions[0].strip()
            timeout = None

            # Check for timeout comment on the first line
            timeout_match = re.match(r"^#\s*timeout:\s*(\d+)\s*\n", action_text)
            if timeout_match:
                requested_timeout = int(timeout_match.group(1))
                # Cap at max_timeout
                timeout = min(requested_timeout, self.config.max_timeout)
                # Remove the timeout comment from the action
                action_text = action_text[timeout_match.end():].strip()

            return {"action": action_text, "timeout": timeout, **response}

        # Create format error message using legacy Jinja2 rendering
        # (since FormatError expects a string exception message)
        template_vars = self.get_template_arguments(actions=actions)
        error_message = Template(self.config.format_error_template, undefined=StrictUndefined).render(**template_vars)
        raise FormatError(error_message)

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"], timeout=action.get("timeout"))
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            template_vars = self.get_template_arguments(action=action, output=output)
            error_message = Template(self.config.timeout_template, undefined=StrictUndefined).render(**template_vars)
            raise ExecutionTimeoutError(error_message)
        except TimeoutError:
            template_vars = self.get_template_arguments(action=action, output="")
            error_message = Template(self.config.timeout_template, undefined=StrictUndefined).render(**template_vars)
            raise ExecutionTimeoutError(error_message)
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))
