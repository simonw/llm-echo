import json
import llm
import llm.models
import pytest


def test_prompt():
    model = llm.get_model("echo")
    response = model.prompt("prompt", system="system")
    assert json.loads(str(response)) == {
        "prompt": "prompt",
        "system": "system",
        "attachments": [],
        "stream": True,
        "previous": [],
    }


def test_prompt_with_option():
    model = llm.get_model("echo")
    response = model.prompt("prompt", system="system", example_bool=True)
    assert json.loads(str(response)) == {
        "prompt": "prompt",
        "system": "system",
        "attachments": [],
        "stream": True,
        "previous": [],
        "options": {"example_bool": True},
    }


def test_conversation():
    model = llm.get_model("echo")
    conversation = model.conversation()
    # str forces the prompt to execute
    str(conversation.prompt("hi1", system="system"))
    response = conversation.prompt("hi2")
    assert json.loads(str(response)) == {
        "prompt": "hi2",
        "system": "",
        "attachments": [],
        "stream": True,
        "previous": [{"prompt": "hi1"}],
    }


@pytest.mark.asyncio
async def test_async_prompt():
    model = llm.get_async_model("echo")
    response = await model.prompt("prompt", system="system")
    text = await response.text()
    assert json.loads(text) == {
        "prompt": "prompt",
        "system": "system",
        "attachments": [],
        "stream": True,
        "previous": [],
    }


def test_prompt_with_tool_calls():
    def example(input: str) -> str:
        return f"Example output for {input}"

    model = llm.get_model("echo")
    chain_response = model.chain(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "name": "example",
                        "arguments": {"input": "test"},
                    }
                ],
                "prompt": "prompt",
            }
        ),
        system="system",
        tools=[example],
    )
    responses = list(chain_response.responses())
    tool_calls = responses[0].tool_calls()
    assert tool_calls == [
        llm.ToolCall(name="example", arguments={"input": "test"}, tool_call_id=None)
    ]
    assert responses[1].prompt.tool_results == [
        llm.models.ToolResult(
            name="example", output="Example output for test", tool_call_id=None
        )
    ]
    assert json.loads(responses[1].text()) == {
        "prompt": "",
        "system": "",
        "attachments": [],
        "stream": True,
        "previous": [
            {
                "prompt": '{"tool_calls": [{"name": "example", "arguments": {"input": "test"}}], "prompt": "prompt"}'
            }
        ],
        "tool_results": [
            {
                "name": "example",
                "output": "Example output for test",
                "tool_call_id": None,
            }
        ],
    }


def test_raw():
    model = llm.get_model("echo")
    response = model.prompt(
        json.dumps(
            {
                "raw": "this is the raw text",
                "misc": "Other stuff",
            }
        ),
        system="system",
    )
    output = response.text()
    assert output == "this is the raw text"
