# LMStudio-Assistant Middleware

This repository provides a small middleware layer that forwards user prompts to
an OpenAI-compatible API (such as LM Studio) while safely executing tool calls
requested by the downstream model.

## Features

* Interactive loop that proxies messages between the user and the model.
* Allow-listed tool execution for `run_python` and a read-only `run_shell`.
* JSON-formatted tool schemas to guide the LLM.
* Structured logging of tool usage for observability.

## Requirements

* Python 3.9+
* [`openai`](https://pypi.org/project/openai/) Python package

## Installation

1. Clone the repository and move into the project directory:

   ```bash
   git clone https://github.com/your-org/LMStudio-Assistant.git
   cd LMStudio-Assistant
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. Install the Python dependencies (the middleware does **not** auto-configure
   them for you). A `requirements.txt` file is included so you can install
   everything with a single command:

   ```bash
   pip install -r requirements.txt
   ```

### macOS-specific notes

* Use the system `python3` binary (or a Homebrew-managed interpreter) when
  creating the virtual environment: `python3 -m venv .venv` and
  `source .venv/bin/activate`.
* Install dependencies with `pip3 install -r requirements.txt` if your
  environment does not alias `pip` to the Python 3 package manager.
* Start LM Studio in server mode ("OpenAI Compatible Server") and make sure it
  listens on `http://localhost:1234/v1` before running the middleware.
* The middleware will not auto-discover or auto-start LM Studio—you must keep
  it running while you interact with the middleware.

## Usage

Set the environment variables for your OpenAI-compatible endpoint (LM Studio,
OpenAI, etc.). These values are not auto-populated—the middleware expects them
to be present in your shell environment:

```bash
export OPENAI_BASE_URL="http://localhost:1234/v1"
export OPENAI_API_KEY="lm-studio"
```

Run the middleware and pass the downstream model name:

```bash
python middleware.py --model lmstudio-community/your-model-name
```

When prompted, enter the first user message. The middleware will continue the
conversation, automatically executing tool calls until the model responds with a
plain assistant message.

To provide the initial message as an argument:

```bash
python middleware.py --model lmstudio-community/your-model-name "Plot y = x^2"
```

Adjust logging verbosity using `--log-level` (e.g., `DEBUG`).

## Tool Policies

* Only the `run_python` and `run_shell` tools are available.
* `run_shell` accepts commands from a small read-only allow list (`ls`, `pwd`,
  `echo`, `cat`, `head`, `tail`). Any other command is rejected.
* Each tool invocation has a 5-second timeout. Timeouts or disallowed usage are
  reported back to the downstream model as structured errors.

## Development

Run a quick sanity check by compiling the sources:

```bash
python -m compileall middleware.py
```

This ensures there are no syntax errors before running the middleware.
