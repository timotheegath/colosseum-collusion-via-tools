"""Scoping tests for the native ``post_message`` restriction.

The old ContextVar-based monkey-patch is gone: ``post_message`` availability is
now driven per phase and per agent by ``environment.blocked_tools`` (mapping
entries with ``roles``/``agents``), read by ``terrarium.tools.gates.tool_blocked``
and honoured both in the schema (toolset discovery bound to the env) and at
runtime (the protocol's tool-call handler). Because the gate is config-driven
rather than task-local state, concurrent runs cannot crosstalk by construction.

No network or LLM call is involved.
"""

from __future__ import annotations

import types
import unittest

from terrarium.tools.discovery import ToolsetDiscovery
from terrarium.tools.gates import tool_blocked


def _env(blocked_tools, role_by_agent=None):
    return types.SimpleNamespace(
        env_config={"blocked_tools": blocked_tools},
        role_by_agent=role_by_agent or {},
        tools_environment_name="HospitalEnvironment",
    )


ADVERSARY_BLOCK = {"planning": [{"tool": "post_message", "roles": ["adversary"]}]}


class PostMessageScopingTests(unittest.TestCase):
    def test_roles_entry_scopes_to_adversaries(self) -> None:
        env = _env(
            ADVERSARY_BLOCK,
            role_by_agent={"AgentA": "adversary", "AgentB": "cooperative"},
        )
        self.assertTrue(tool_blocked(env, "post_message", "planning", "AgentA"))
        self.assertFalse(tool_blocked(env, "post_message", "planning", "AgentB"))

    def test_agents_entry_scopes_to_named_agents(self) -> None:
        env = _env(
            {"planning": [{"tool": "post_message", "agents": ["AgentA"]}]},
            role_by_agent={"AgentA": "adversary", "AgentB": "adversary"},
        )
        self.assertTrue(tool_blocked(env, "post_message", "planning", "AgentA"))
        self.assertFalse(tool_blocked(env, "post_message", "planning", "AgentB"))

    def test_gate_is_per_env_not_global(self) -> None:
        # Two concurrent "runs" with different configs cannot crosstalk: the
        # gate reads each env's own env_config.
        blocked = _env(ADVERSARY_BLOCK, {"AgentA": "adversary"})
        open_env = _env({}, {"AgentA": "adversary"})
        self.assertTrue(tool_blocked(blocked, "post_message", "planning", "AgentA"))
        self.assertFalse(tool_blocked(open_env, "post_message", "planning", "AgentA"))

    def test_discovery_withholds_schema_per_agent(self) -> None:
        env = _env(
            ADVERSARY_BLOCK,
            role_by_agent={"AgentA": "adversary", "AgentB": "cooperative"},
        )
        discovery = ToolsetDiscovery()
        discovery.bind_environment(env)
        adv_names = {
            t["function"]["name"]
            for t in discovery.get_tools_for_blackboard("planning", "HospitalEnvironment", agent_name="AgentA")
        }
        coop_names = {
            t["function"]["name"]
            for t in discovery.get_tools_for_blackboard("planning", "HospitalEnvironment", agent_name="AgentB")
        }
        self.assertNotIn("post_message", adv_names)
        self.assertIn("post_message", coop_names)


if __name__ == "__main__":
    unittest.main()
