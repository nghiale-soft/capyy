from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.routes.messages import router


class _Settings:
    local_api_key = None


class _History:
    def resolve_project(self, request, body):
        return "project-test"

    def record_messages(self, *args, **kwargs):
        pass

    def build_context(self, *args, **kwargs):
        return ""

    def record(self, *args, **kwargs):
        pass


class _GenericGateway:
    def __init__(self) -> None:
        self.payload = None

    def resolve(self, model):
        return "priority-one", "generic-model"

    def is_freebuff(self, provider_id):
        return False

    async def chat(self, provider_id, payload, *, real_model=None):
        self.payload = payload
        return {
            "id": "chatcmpl-test",
            "model": real_model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"path":"a.txt"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }


class AnthropicMessagesRoutingTests(unittest.TestCase):
    def test_generic_provider_preserves_client_tool_contract(self) -> None:
        app = FastAPI()
        app.include_router(router)
        app.state.settings = _Settings()
        app.state.gateway = _GenericGateway()
        app.state.accounts = object()
        app.state.chat_history = _History()

        with TestClient(app) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "generic-model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Read a.txt"}],
                    "tools": [
                        {
                            "name": "Read",
                            "description": "Read a file",
                            "input_schema": {"type": "object"},
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("tools", app.state.gateway.payload)
        block = response.json()["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["name"], "Read")
        self.assertEqual(block["input"], {"path": "a.txt"})


if __name__ == "__main__":
    unittest.main()
