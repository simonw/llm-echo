import llm
from llm import parts
import json
from typing import Optional
from typing import AsyncGenerator
from pydantic import Field

THINKING_CHUNKS = ("First I consider the prompt, ", "then I decide what to say.")


@llm.hookimpl
def register_models(register):
    register(Echo(), EchoAsync())
    register(EchoNeedsKey(), EchoNeedsKeyAsync())


class _Shared:
    model_id = "echo"
    can_stream = True
    supports_tools = True
    attachment_types = ("image/png", "image/jpeg", "image/gif")

    class Options(llm.Options):
        example_bool: Optional[bool] = Field(
            description="Example boolean option",
            default=None,
        )
        thinking: Optional[bool] = Field(
            description="If set, emit canned reasoning chunks before the response",
            default=None,
        )

    def shared(self, prompt, stream, response, conversation):
        prompt_text = prompt.prompt
        raw = None
        # Pending tool_call StreamEvents the caller should yield before the
        # text part. Emitting them as StreamEvents (rather than the legacy
        # response.add_tool_call API) means they appear in
        # response.to_dict()["messages"] as tool_call Parts, matching what
        # real model adapters do.
        self._pending_tool_call_events = []
        if prompt_text.strip() and prompt_text.strip()[0] == "{":
            try:
                prompt_dict = json.loads(prompt_text)
                raw = prompt_dict.get("raw", None)
                prompt_text = prompt_dict.get("prompt", "")
                tool_calls = prompt_dict.get("tool_calls", [])
                for idx, tool_call in enumerate(tool_calls):
                    name = tool_call["name"]
                    arguments = tool_call.get("arguments") or {}
                    # llm 0.32 keeps two parallel stores: _stream_events
                    # (read by to_dict / response.messages) and _tool_calls
                    # (read by the chain framework's execute_tool_calls).
                    # Populate BOTH — yielding StreamEvents alone makes
                    # tool calls visible to persistence but invisible to
                    # the chain loop, so the chain stops after one
                    # response.
                    response.add_tool_call(llm.ToolCall(name=name, arguments=arguments))
                    self._pending_tool_call_events.append(
                        parts.StreamEvent(
                            type="tool_call_name",
                            chunk=name,
                            part_index=10 + idx,
                            tool_name=name,
                        )
                    )
                    self._pending_tool_call_events.append(
                        parts.StreamEvent(
                            type="tool_call_args",
                            chunk=json.dumps(arguments),
                            part_index=10 + idx,
                        )
                    )
            except json.JSONDecodeError:
                pass

        if raw is not None:
            return raw

        info = {
            "prompt": prompt_text,
            "system": prompt.system,
            "attachments": [
                {"type": a.type, "path": a.path, "url": a.url, "id": a.id()}
                for a in prompt.attachments
            ],
            "stream": stream,
            "previous": (
                [
                    {
                        "prompt": r.prompt.prompt,
                    }
                    for r in conversation.responses
                ]
                if conversation
                else []
            ),
        }
        if prompt.options.example_bool is not None:
            info["options"] = {
                "example_bool": prompt.options.example_bool,
            }
        if prompt.tool_results:
            info["tool_results"] = [
                {
                    "name": r.name,
                    "output": r.output,
                    "tool_call_id": r.tool_call_id,
                }
                for r in prompt.tool_results
            ]
        return info


class Echo(_Shared, llm.Model):
    def execute(self, prompt, stream, response, conversation=None):
        data = self.shared(prompt, stream, response, conversation)
        if isinstance(data, dict):
            output = json.dumps(data, indent=2)
        else:
            output = data
        response.set_usage(input=len(prompt.prompt.split()), output=len(output.split()))
        if prompt.options.thinking:
            for chunk in THINKING_CHUNKS:
                yield parts.StreamEvent(type="reasoning", chunk=chunk, part_index=0)
            yield parts.StreamEvent(type="text", chunk=output, part_index=1)
            for ev in self._pending_tool_call_events:
                yield ev
        else:
            yield parts.StreamEvent(type="text", chunk=output, part_index=1)
            for ev in self._pending_tool_call_events:
                yield ev


class EchoAsync(_Shared, llm.AsyncModel):
    async def execute(
        self, prompt, stream, response, conversation=None
    ) -> AsyncGenerator[str, None]:
        data = self.shared(prompt, stream, response, conversation)
        if isinstance(data, dict):
            output = json.dumps(data, indent=2)
        else:
            output = data
        response.set_usage(input=len(prompt.prompt.split()), output=len(output.split()))
        if prompt.options.thinking:
            for chunk in THINKING_CHUNKS:
                yield parts.StreamEvent(type="reasoning", chunk=chunk, part_index=0)
            yield parts.StreamEvent(type="text", chunk=output, part_index=1)
            for ev in self._pending_tool_call_events:
                yield ev
        else:
            yield parts.StreamEvent(type="text", chunk=output, part_index=1)
            for ev in self._pending_tool_call_events:
                yield ev


class _SharedNeedsKey(_Shared):
    model_id = "echo-needs-key"
    needs_key = "echo-needs-key"
    key_env_var = "LLM_ECHO_NEEDS_KEY_KEY"


class EchoNeedsKey(_SharedNeedsKey, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation=None, key=None):
        data = self.shared(prompt, stream, response, conversation)
        if isinstance(data, dict):
            data["key"] = key
            output = json.dumps(data, indent=2)
        else:
            output = data
        response.set_usage(input=len(prompt.prompt.split()), output=len(output.split()))
        if prompt.options.thinking:
            for chunk in THINKING_CHUNKS:
                yield parts.StreamEvent(type="reasoning", chunk=chunk, part_index=0)
            yield parts.StreamEvent(type="text", chunk=output, part_index=1)
            for ev in self._pending_tool_call_events:
                yield ev
        else:
            yield parts.StreamEvent(type="text", chunk=output, part_index=1)
            for ev in self._pending_tool_call_events:
                yield ev


class EchoNeedsKeyAsync(_SharedNeedsKey, llm.AsyncKeyModel):
    async def execute(
        self, prompt, stream, response, conversation=None, key=None
    ) -> AsyncGenerator[str, None]:
        data = self.shared(prompt, stream, response, conversation)
        if isinstance(data, dict):
            data["key"] = key
            output = json.dumps(data, indent=2)
        else:
            output = data
        response.set_usage(input=len(prompt.prompt.split()), output=len(output.split()))
        if prompt.options.thinking:
            for chunk in THINKING_CHUNKS:
                yield parts.StreamEvent(type="reasoning", chunk=chunk, part_index=0)
            yield parts.StreamEvent(type="text", chunk=output, part_index=1)
            for ev in self._pending_tool_call_events:
                yield ev
        else:
            yield parts.StreamEvent(type="text", chunk=output, part_index=1)
            for ev in self._pending_tool_call_events:
                yield ev
