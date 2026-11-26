"""Per-run scoping tests for the ``post_message`` restriction.

These tests exercise the *native* per-phase tool policy on
``terrarium.communication_protocols.local.LocalCommunicationProtocol``
(``set_phase_policy`` / ``pop_phase_policy`` / ``active_phase_policy``) and
``terrarium.tools.discovery.ToolsetDiscovery``'s ``allowed_tools`` filter
directly. No network or LLM call is involved.

Scoping note (why per-instance data is sufficient here): one protocol instance
is created per run (``run.py`` builds ``LocalCommunicationProtocol(config=cfg)``
inside ``_run_single``), so the ``_phase_policies`` dict on the instance is
per-run -- concurrent ``_run_single`` coroutines scheduled by a sweep
(``max_concurrent_runs > 1``) each own their protocol and can never clobber one
another's restriction. Within a run the policy is keyed by phase label, and the
per-turn tool build reads it via ``BaseAgent.set_meta_context``.
"""

from __future__ import annotations

import asyncio
import unittest

from terrarium.communication_protocols.local import (
    LocalCommunicationProtocol,
    PhaseToolPolicy,
)
from terrarium.tools.discovery import ToolsetDiscovery


def _withheld_policy() -> PhaseToolPolicy:
    return PhaseToolPolicy(blackboard_tools=set())


class PostMessageScopingTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_runs_do_not_crosstalk(self) -> None:
        a_wrote = asyncio.Event()
        b_wrote = asyncio.Event()

        async def task_a() -> set:
            proto = LocalCommunicationProtocol(config={})
            proto.set_phase_policy("planning", _withheld_policy())
            a_wrote.set()
            await b_wrote.wait()
            policy = proto.active_phase_policy("planning")
            return set(policy.blackboard_tools) if policy is not None else None

        async def task_b() -> set:
            proto = LocalCommunicationProtocol(config={})
            proto.set_phase_policy(
                "planning", PhaseToolPolicy(blackboard_tools={"post_message"})
            )
            b_wrote.set()
            await a_wrote.wait()
            policy = proto.active_phase_policy("planning")
            return set(policy.blackboard_tools) if policy is not None else None

        observed_a, observed_b = await asyncio.gather(task_a(), task_b())

        # Each run reads back its own policy, not the other run's.
        self.assertEqual(observed_a, set())
        self.assertEqual(observed_b, {"post_message"})
        # ...and neither run's policy leaked into a fresh protocol.
        fresh = LocalCommunicationProtocol(config={})
        self.assertIsNone(fresh.active_phase_policy("planning"))

    async def test_policy_is_scoped_per_phase(self) -> None:
        proto = LocalCommunicationProtocol(config={})

        async def withheld_task() -> tuple:
            proto.set_phase_policy("planning", _withheld_policy())
            # Yield so the other task interleaves with the policy set.
            await asyncio.sleep(0)
            discovery = ToolsetDiscovery()
            policy = proto.active_phase_policy("planning")
            return (
                discovery.get_tools_for_blackboard(
                    "planning", allowed_tools=policy.blackboard_tools
                ),
                discovery.get_blackboard_tool_names(allowed_tools=policy.blackboard_tools),
                proto.active_phase_policy("pre_coordination"),
            )

        async def default_task() -> tuple:
            # Runs against the same protocol but a phase with no policy: must
            # NOT observe the other phase's withholding.
            await asyncio.sleep(0)
            discovery = ToolsetDiscovery()
            policy = proto.active_phase_policy("pre_coordination")
            allowed = policy.blackboard_tools if policy is not None else None
            return (
                discovery.get_tools_for_blackboard("pre_coordination", allowed_tools=allowed),
                discovery.get_blackboard_tool_names(allowed_tools=allowed),
            )

        (d_tools, d_names, other_phase_policy), (e_tools, e_names) = await asyncio.gather(
            withheld_task(), default_task()
        )

        self.assertEqual(d_tools, [])
        self.assertEqual(d_names, set())
        # The policy is keyed by phase: "pre_coordination" (not planning-shaped
        # by default) is unaffected by the "planning" policy.
        self.assertIsNone(other_phase_policy)
        self.assertEqual(e_tools, [])
        self.assertEqual(e_names, {"post_message"})


class PostMessageDiscoveryTests(unittest.TestCase):
    def test_default_context_returns_original_tools(self) -> None:
        discovery = ToolsetDiscovery()
        tools = discovery.get_tools_for_blackboard("planning")
        self.assertTrue(tools)
        self.assertEqual(tools[0]["function"]["name"], "post_message")
        self.assertEqual(discovery.get_blackboard_tool_names(), {"post_message"})

    def test_withheld_policy_returns_empty(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        proto.set_phase_policy("planning", _withheld_policy())
        policy = proto.active_phase_policy("planning")
        discovery = ToolsetDiscovery()
        self.assertEqual(
            discovery.get_tools_for_blackboard(
                "planning", allowed_tools=policy.blackboard_tools
            ),
            [],
        )
        self.assertEqual(
            discovery.get_tools_for_blackboard(
                "execution", allowed_tools=policy.blackboard_tools
            ),
            [],
        )
        self.assertEqual(
            discovery.get_blackboard_tool_names(allowed_tools=policy.blackboard_tools),
            set(),
        )

    def test_toggle_restores_original_tools(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        discovery = ToolsetDiscovery()
        proto.set_phase_policy("planning", _withheld_policy())
        policy = proto.active_phase_policy("planning")
        self.assertEqual(
            discovery.get_tools_for_blackboard(
                "planning", allowed_tools=policy.blackboard_tools
            ),
            [],
        )
        proto.pop_phase_policy("planning")
        self.assertIsNone(proto.active_phase_policy("planning"))
        self.assertTrue(discovery.get_tools_for_blackboard("planning"))
        self.assertEqual(discovery.get_blackboard_tool_names(), {"post_message"})


if __name__ == "__main__":
    unittest.main()
