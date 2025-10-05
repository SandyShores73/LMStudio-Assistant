"""Middleware bridge for routing user prompts to an OpenAI-compatible LLM.

This module implements the "middleware mode" workflow described in the task:

* Forward user messages to a downstream LLM via an OpenAI-compatible API.
* Detect and execute tool calls produced by the model (currently an allow list
  of ``run_python`` and a limited ``run_shell``).
* Return tool execution results back to the model, repeating until the model
  replies with a regular assistant message.
* Keep the execution sandboxed with short timeouts and without leaking the
  internal routing details to the user.

The module exposes a :class:`Middleware` class with a ``chat`` method that can
be embedded into other applications as well as a small interactive CLI for
manual testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableSequence, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - import guard for runtime
    raise SystemExit(
        "The 'openai' package is required to run the middleware. Install it via\n"
        "  pip install openai"
    ) from exc


# ---------------------------------------------------------------------------
# Data structures


@dataclass
class ToolExecutionResult:
    """Structured response returned to the LLM after running a tool."""

    stdout: str
    stderr: str
    exit_code: int

    def to_json(self) -> str:
        return json.dumps({
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
        })


class ToolExecutionError(Exception):
    """Raised when a tool invocation fails due to invalid input or runtime issues."""


# ---------------------------------------------------------------------------
# Tool implementations


class ToolRegistry:
    """Keeps track of available tools and their execution handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Mapping[str, object]], ToolExecutionResult]] = {}

    def register(
        self,
        name: str,
        handler: Callable[[Mapping[str, object]], ToolExecutionResult],
    ) -> None:
        if name in self._handlers:
            raise ValueError(f"Tool '{name}' already registered")
        self._handlers[name] = handler

    def execute(self, name: str, arguments: Mapping[str, object]) -> ToolExecutionResult:
        if name not in self._handlers:
            raise ToolExecutionError(f"Tool '{name}' is not permitted")
        return self._handlers[name](arguments)


class ToolRunner:
    """Executes allow-listed tools in a sandboxed fashion."""

    PYTHON_TIMEOUT_SEC = 5
    SHELL_TIMEOUT_SEC = 5

    #: Commands permitted for the ``run_shell`` tool. The commands are executed
    #: without a shell, so each entry should be the binary name only.
    SHELL_ALLOW_LIST: Iterable[str] = (
        "ls",
        "pwd",
        "echo",
        "cat",
        "head",
        "tail",
    )

    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register("run_python", self._run_python)
        self.registry.register("run_shell", self._run_shell)

    # -- individual tool implementations -------------------------------------------------

    def _run_python(self, arguments: Mapping[str, object]) -> ToolExecutionResult:
        code = arguments.get("code")
        if not isinstance(code, str):
            raise ToolExecutionError("'code' must be a string")

        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=self.PYTHON_TIMEOUT_SEC,
        )
        return ToolExecutionResult(
            stdout=process.stdout,
            stderr=process.stderr,
            exit_code=process.returncode,
        )

    def _run_shell(self, arguments: Mapping[str, object]) -> ToolExecutionResult:
        command = arguments.get("command")
        if not isinstance(command, str):
            raise ToolExecutionError("'command' must be a string")

        parts = shlex.split(command)
        if not parts:
            raise ToolExecutionError("'command' must not be empty")

        binary = parts[0]
        if binary not in self.SHELL_ALLOW_LIST:
            raise ToolExecutionError(
                f"Command '{binary}' is not permitted by the middleware"
            )

        process = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=self.SHELL_TIMEOUT_SEC,
        )
        return ToolExecutionResult(
            stdout=process.stdout,
            stderr=process.stderr,
            exit_code=process.returncode,
        )


# ---------------------------------------------------------------------------
# Middleware logic


class Middleware:
    """Interactive middleware coordinating between the user and an LLM."""

    SYSTEM_PROMPT = (
        "You are Codex operating in middleware mode. Follow the workflow strictly: "
        "pass user messages to the downstream LLM, execute tool calls using the "
        "provided tools, and never expose internal routing details to the user."
    )

    TOOLS_SCHEMA = [
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": (
                    "Execute Python code in a restricted interpreter. Return stdout, "
                    "stderr, and exit code."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source code to execute.",
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": (
                    "Execute a read-only shell command from an allow list."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command (e.g. 'ls -l').",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    def __init__(
        self,
        client: OpenAI,
        model: str,
        *,
        tool_runner: Optional[ToolRunner] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._tool_runner = tool_runner or ToolRunner()
        self._logger = logger or logging.getLogger(__name__)

    # -- public API ---------------------------------------------------------------------

    def chat(self, user_message: str, *, history: Optional[MutableSequence[Mapping[str, str]]] = None) -> str:
        """Process a single user message, returning the LLM's final reply."""

        conversation: List[Mapping[str, str]] = history.copy() if history else []
        conversation.insert(0, {"role": "system", "content": self.SYSTEM_PROMPT})
        conversation.append({"role": "user", "content": user_message})

        while True:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=conversation,
                tools=self.TOOLS_SCHEMA,
            )

            choice = response.choices[0]
            message = choice.message

            if message.tool_calls:
                for call in message.tool_calls:
                    result_content = self._handle_tool_call(call.function.name, call.function.arguments)
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result_content,
                        }
                    )
                continue

            if message.content is None:
                raise RuntimeError("Received empty assistant message without tool calls")

            return message.content

    # -- helpers ------------------------------------------------------------------------

    def _handle_tool_call(self, name: str, arguments_json: str) -> str:
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"Invalid JSON arguments for tool '{name}': {exc}")

        self._logger.info("Executing tool", extra={"tool_name": name, "arguments": arguments})

        try:
            result = self._tool_runner.registry.execute(name, arguments)
        except subprocess.TimeoutExpired:
            result_content = json.dumps({
                "error": f"Tool '{name}' timed out after execution limit.",
            })
        except ToolExecutionError as exc:
            result_content = json.dumps({"error": str(exc)})
        except Exception as exc:  # pragma: no cover - safety net
            result_content = json.dumps({"error": f"Unexpected error: {exc}"})
        else:
            result_content = result.to_json()

        self._logger.info("Tool execution complete", extra={"tool_name": name, "result": result_content})
        return result_content


# ---------------------------------------------------------------------------
# Command-line interface


def build_client(base_url: Optional[str], api_key: Optional[str]) -> OpenAI:
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Codex middleware loop")
    parser.add_argument("message", nargs="?", help="Initial user message. If omitted, read from stdin interactively.")
    parser.add_argument("--model", required=True, help="Downstream model identifier")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="Base URL for the OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="API key for the endpoint")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args(argv)


def interactive_prompt() -> str:
    try:
        return input("User: ")
    except EOFError:  # pragma: no cover - CLI nicety
        return ""


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    client = build_client(args.base_url, args.api_key)
    middleware = Middleware(client, args.model)

    user_message = args.message or interactive_prompt()
    if not user_message:
        raise SystemExit("No user message provided")

    response = middleware.chat(user_message)
    print(f"Assistant: {response}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
