"""Tests that the provisioner sees the blackboard natively.

The blind-provisioner bug is fixed upstream in the Terrarium submodule
(``fix/blind_provisioner``): ``HospitalPrompts.get_user_prompt`` now renders
the BLACKBOARD section inside the ``Resource_Provisioner`` branch too, so
Colosseum's ``ProvisionerBlackboardPrompts`` wrapper was deleted. These tests
pin the native behavior so the wrapper is not reintroduced.

Pure stdlib ``unittest``: no network, no LLM, no environment construction.
"""

from __future__ import annotations

import types
import unittest

from terrarium.environments.dcops.hospital.hospital_prompts import HospitalPrompts

TWO_BOARDS = {0: "PUBLIC board body", 1: "SECRET board body"}


def _prompts() -> HospitalPrompts:
    env = types.SimpleNamespace(persona_by_agent={})
    return HospitalPrompts(env, {})


def _provisioner_ctx() -> dict:
    return {
        "inventory_snapshot": {"General_Hospital": {"IV_Kits": 5}},
        "failures_by_hospital": {},
        "cost_table": [],
        "phase": "planning",
    }


def _dept_ctx() -> dict:
    return {
        "dept_info": {"capacity": 2, "service": "Triage", "hospital": "General_Hospital"},
        "job_queue": [],
        "inventory_snapshot": {},
        "phase": "planning",
    }


class ProvisionerSeesBlackboardTests(unittest.TestCase):
    def test_provisioner_prompt_contains_both_boards(self) -> None:
        prompts = _prompts()
        prompt = prompts.get_user_prompt("Resource_Provisioner", _provisioner_ctx(), TWO_BOARDS)
        self.assertIn("=== BLACKBOARD (ID: 0) ===", prompt)
        self.assertIn("[0]", prompt)
        self.assertIn("[1]", prompt)
        self.assertIn("PUBLIC board body", prompt)
        self.assertIn("SECRET board body", prompt)

    def test_provisioner_prompt_keeps_role_and_instructions(self) -> None:
        prompts = _prompts()
        prompt = prompts.get_user_prompt("Resource_Provisioner", _provisioner_ctx(), TWO_BOARDS)
        self.assertIn("Role: Logistics Provisioner", prompt)
        self.assertIn("Check the Blackboard", prompt)

    def test_department_prompt_still_contains_blackboard(self) -> None:
        prompts = _prompts()
        prompt = prompts.get_user_prompt("General_Hospital_Triage", _dept_ctx(), TWO_BOARDS)
        self.assertIn("=== BLACKBOARD (ID: 0) ===", prompt)
        self.assertIn("PUBLIC board body", prompt)

    def test_coalition_block_rendered_for_adversary_provisioner(self) -> None:
        prompts = _prompts()
        ctx = _provisioner_ctx()
        ctx.update(
            {
                "role": "adversary",
                "coalition_members": ["Resource_Provisioner", "General_Hospital_Triage"],
                "collusion_strategy": "covert",
                "collusion_instructions": "SECRET INSTRUCTIONS BODY",
                "secret_blackboard_id": 1,
            }
        )
        prompt = prompts.get_user_prompt("Resource_Provisioner", ctx, TWO_BOARDS)
        self.assertIn("COALITION INSTRUCTIONS", prompt)
        self.assertIn("SECRET INSTRUCTIONS BODY", prompt)
        self.assertIn("Secret coordination blackboard ID: 1", prompt)

    def test_coalition_block_hidden_from_cooperative_agents(self) -> None:
        prompts = _prompts()
        ctx = _provisioner_ctx()
        ctx.update(
            {
                "role": "cooperative",
                "coalition_members": ["Resource_Provisioner"],
                "collusion_instructions": "SECRET INSTRUCTIONS BODY",
                "secret_blackboard_id": 1,
            }
        )
        prompt = prompts.get_user_prompt("Resource_Provisioner", ctx, TWO_BOARDS)
        self.assertNotIn("COALITION INSTRUCTIONS", prompt)
        self.assertNotIn("SECRET INSTRUCTIONS BODY", prompt)


if __name__ == "__main__":
    unittest.main()
