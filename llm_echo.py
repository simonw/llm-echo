import llm
import json
from typing import AsyncGenerator


@llm.hookimpl
def register_models(register):
    register(Echo(), EchoAsync())


class _Shared:
    model_id = "echo"
    can_stream = True
    attachment_types = ("image/png", "image/jpeg", "image/gif")

    def shared(self, prompt, stream, response, conversation):
        return {
            "prompt": prompt.prompt,
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


class Echo(_Shared, llm.Model):
    def execute(self, prompt, stream, response, conversation=None):
        data = self.shared(prompt, stream, response, conversation)
        yield json.dumps(data, indent=2)


class EchoAsync(_Shared, llm.AsyncModel):
    async def execute(
        self, prompt, stream, response, conversation=None
    ) -> AsyncGenerator[str, None]:
        data = self.shared(prompt, stream, response, conversation)
        yield json.dumps(data, indent=2)
