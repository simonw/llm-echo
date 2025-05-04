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
