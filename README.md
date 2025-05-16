# llm-echo

[![PyPI](https://img.shields.io/pypi/v/llm-echo.svg)](https://pypi.org/project/llm-echo/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-echo?include_prereleases&label=changelog)](https://github.com/simonw/llm-echo/releases)
[![Tests](https://github.com/simonw/llm-echo/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-echo/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-echo/blob/main/LICENSE)

Debug plugin for LLM. Adds a model which echos its input without hitting an API or executing a local LLM.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-echo
```
## Usage

The plugin adds a `echo` model which simply echos the prompt details back to you as JSON.

```bash
llm -m echo prompt -s 'system prompt'
```
Output:
```json
{
  "prompt": "prompt",
  "system": "system prompt",
  "attachments": [],
  "stream": true,
  "previous": []
}
```
You can also add one example option like this:
```bash
llm -m echo prompt -o example_bool 1
```
Output:
```json
{
  "prompt": "prompt",
  "system": "",
  "attachments": [],
  "stream": true,
  "previous": [],
  "options": {
    "example_bool": true
  }
}
```

## Tool calling

You can use `llm-echo` to test tool calling without needing to run prompts through an actual LLM. In your prompt, send something like this:

```json
{
  "prompt": "This will be treated as the prompt",
  "tool_calls": {
    "name": "example",
    "arguments": {
      "input": "Hello, world!"
    }
  ]
}
```
This will be converted into [tool call requests](https://llm.datasette.io/en/latest/python-api.html#tools) in the response.

Take a look at the [test suite for llm-tools-simpleeval](https://github.com/simonw/llm-tools-simpleeval/blob/main/tests/test_tools_simpleeval.py) for an example of how to use this.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-echo
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
python -m pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
