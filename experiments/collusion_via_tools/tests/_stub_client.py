"""Deterministic stub LLM client for end-to-end smoke tests.

No network: ``generate_response`` returns a canned assistant message. On the
first call per client it emits one ``post_message`` tool call (with a PUBLIC
blackboard_id, so the native secret-channel rewrite is observable); later calls
return plain content. ``process_tool_calls`` mirrors the real clients' tool
loop so the protocol's tool-call handler (gating + rewrite + caps) runs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


class StubLLMClient:
    # Which blackboard the canned post_message targets. Tests may override this
    # class attribute (e.g. point it at the secret board, id 1, when the run
    # creates one).
    blackboard_id = 0

    def __init__(self, agent_name: str = "") -> None:
        self.agent_name = agent_name
        self.calls = 0

    @staticmethod
    def init_context(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def get_external_tools(self) -> List[Dict[str, Any]]:
        return []

    async def get_external_tool_names(self) -> set:
        return set()

    @staticmethod
    def get_usage(response: Dict[str, Any], current_usage: Dict[str, int]) -> Dict[str, int]:
        usage = (response or {}).get("usage") or {}
        current_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        current_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        current_usage["total_tokens"] = (
            current_usage["prompt_tokens"] + current_usage["completion_tokens"]
        )
        return current_usage

    def generate_response(
        self, input: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        self.calls += 1
        message: Dict[str, Any] = {"role": "assistant", "content": "stub response"}
        if self.calls == 1:
            # Deliberately targets the PUBLIC board (id 0): during an enforced
            # secret-channel phase the protocol must rewrite it.
            message["tool_calls"] = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "post_message",
                        "arguments": json.dumps(
                            {
                                "blackboard_id": type(self).blackboard_id,
                                "message": f"secret plan from {self.agent_name}",
                            }
                        ),
                    },
                }
            ]
            message["content"] = ""
        response = {
            "id": "stub",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if self.calls == 1 else "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return response, message.get("content") or ""

    async def process_tool_calls(
        self, response: Dict[str, Any], context: List[Dict[str, Any]], execute_tool_callback: Any
    ) -> Tuple[int, List[Dict[str, Any]], List[str]]:
        choices = response.get("choices") or []
        if not choices:
            return 0, context, []
        message = dict(choices[0].get("message") or {})
        context.append(message)
        executed = 0
        step_tools: List[str] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name") or "unknown_tool"
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, ValueError):
                args = {}
            result = await execute_tool_callback(name, args)
            context.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": json.dumps(result),
                }
            )
            step_tools.append(f"{name} -- {json.dumps(args)}")
            executed += 1
        return executed, context, step_tools
