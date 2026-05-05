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

    def hello() -> str:
        return "Hello world"

    model = llm.get_model("echo")
    chain_response = model.chain(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "name": "example",
                        "arguments": {"input": "test"},
                    },
                    {"name": "hello"},
                ],
                "prompt": "prompt",
            }
        ),
        system="system",
        tools=[example, hello],
    )
    responses = list(chain_response.responses())
    tool_calls = responses[0].tool_calls()
    assert tool_calls == [
        llm.ToolCall(name="example", arguments={"input": "test"}, tool_call_id=None),
        llm.ToolCall(name="hello", arguments={}, tool_call_id=None),
    ]
    assert responses[1].prompt.tool_results == [
        llm.models.ToolResult(
            name="example", output="Example output for test", tool_call_id=None
        ),
        llm.models.ToolResult(name="hello", output="Hello world", tool_call_id=None),
    ]
    assert json.loads(responses[1].text()) == {
        "prompt": "",
        "system": "system",
        "attachments": [],
        "stream": True,
        "previous": [
            {
                "prompt": '{"tool_calls": [{"name": "example", "arguments": {"input": "test"}}, {"name": "hello"}], "prompt": "prompt"}'
            }
        ],
        "tool_results": [
            {
                "name": "example",
                "output": "Example output for test",
                "tool_call_id": None,
            },
            {"name": "hello", "output": "Hello world", "tool_call_id": None},
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


def test_usage():
    model = llm.get_model("echo")
    response = model.prompt("one two three")
    response.text()
    assert response.input_tokens == 3
    assert response.output_tokens > 0


@pytest.mark.asyncio
async def test_usage_async():
    model = llm.get_async_model("echo")
    response = await model.prompt("one two three")
    await response.text()
    assert response.input_tokens == 3
    assert response.output_tokens > 0


def test_echo_needs_key():
    model = llm.get_model("echo-needs-key")
    assert model.needs_key == "echo-needs-key"
    assert model.key_env_var == "LLM_ECHO_NEEDS_KEY_KEY"
    model.key = "sk-test-key-1234"
    response = model.prompt("hello", system="system")
    data = json.loads(response.text())
    assert data == {
        "prompt": "hello",
        "system": "system",
        "attachments": [],
        "stream": True,
        "previous": [],
        "key": "sk-test-key-1234",
    }


def test_echo_needs_key_different_keys():
    """Different calls can use different keys."""
    model = llm.get_model("echo-needs-key")
    model.key = "sk-key-alpha"
    data1 = json.loads(model.prompt("first").text())
    assert data1["key"] == "sk-key-alpha"

    model.key = "sk-key-beta"
    data2 = json.loads(model.prompt("second").text())
    assert data2["key"] == "sk-key-beta"


@pytest.mark.asyncio
async def test_echo_needs_key_async():
    model = llm.get_async_model("echo-needs-key")
    assert model.needs_key == "echo-needs-key"
    model.key = "sk-async-test-5678"
    response = await model.prompt("async hello")
    data = json.loads(await response.text())
    assert data["key"] == "sk-async-test-5678"
    assert data["prompt"] == "async hello"


def test_echo_needs_key_with_raw():
    """Raw mode should still work and not include the key."""
    model = llm.get_model("echo-needs-key")
    model.key = "sk-test-key"
    response = model.prompt(json.dumps({"raw": "raw output"}))
    assert response.text() == "raw output"


def test_echo_needs_key_conversation():
    model = llm.get_model("echo-needs-key")
    model.key = "sk-conv-key"
    conversation = model.conversation()
    str(conversation.prompt("turn1"))
    response = conversation.prompt("turn2")
    data = json.loads(response.text())
    assert data["key"] == "sk-conv-key"
    assert data["previous"] == [{"prompt": "turn1"}]


def test_thinking_option():
    model = llm.get_model("echo")
    response = model.prompt("hi", thinking=True)
    response.text()
    parts = response.messages()[0].parts
    assert [type(p).__name__ for p in parts] == ["ReasoningPart", "TextPart"]
    assert parts[0].text == "First I consider the prompt, then I decide what to say."
    assert json.loads(parts[1].text)["prompt"] == "hi"


@pytest.mark.asyncio
async def test_thinking_option_async():
    model = llm.get_async_model("echo")
    response = await model.prompt("hi", thinking=True)
    events = []
    async for event in response.astream_events():
        events.append(event)
    reasoning = [e for e in events if e.type == "reasoning"]
    assert [e.chunk for e in reasoning] == [
        "First I consider the prompt, ",
        "then I decide what to say.",
    ]
