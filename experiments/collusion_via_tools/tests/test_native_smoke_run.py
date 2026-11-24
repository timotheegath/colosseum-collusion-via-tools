"""End-to-end smoke test of the native config-driven wiring.

Runs ``run_from_config`` on a tiny config with a stub LLM client (no network)
and asserts the native behavior survived the refactor:

* ``agent_turns.json`` contains pre_coordination (iteration 0, adversaries
  only) + planning + execution + survey phases;
* the coalition-only secret blackboard exists with exactly the coalition as
  participants, and the stub's public-board ``post_message`` was rewritten onto
  it during pre_coordination (enforce_in_phases);
* adversaries' ``post_message`` is blocked in planning/execution/survey via
  ``environment.blocked_tools`` (tool events show ``blocked`` results);
* ``run_config.json`` records the secret_blackboard_id.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import unittest.mock
import yaml

from experiments.collusion_via_tools import run as run_mod
from experiments.collusion_via_tools.tests._stub_client import StubLLMClient

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

_SMOKE_CONFIG = {
    "experiment": {
        "tag": "smoke",
        "max_concurrent_runs": 1,
        "seeds": [0],
        "adversary": {
            "attack_frequency": "always",
            "secret_channel_enabled": True,
            "instructions_file": str(_CONFIGS_DIR / "collusion_instructions.txt"),
        },
        "sweeps": [
            {
                "name": "smoke_sweep",
                "topologies": ["complete"],
                "adversary_counts": [2],
                "target_roles": ["Resource_Provisioner"],
                "num_agents": [5],
                "num_patients": ["num_agents_minus_1"],
                "disable_post_message": [True],
                "pre_coordination_round": True,
                "pre_coordination_rounds": [1],
            }
        ],
    },
    "simulation": {
        "seed": 42,
        "max_iterations": 1,
        "max_planning_rounds": 1,
        "max_conversation_steps": 2,
    },
    "environment": {
        "import_path": "terrarium.environments.dcops.hospital.hospital_env:HospitalEnvironment",
        "name": "HospitalEnvironment",
        "num_patients": 4,
        "num_hospitals": 1,
        "default_capacity": 2,
    },
    "communication_network": {
        "topology": "complete",
        "num_agents": 5,
        "consolidate_channels": True,
        "center": 0,
    },
    "llm_models": [
        {
            "label": "stub",
            "llm": {
                "provider": "openai",
                "openai": {"model": "stub-model", "params": {"max_tokens": 128, "temperature": 0.0}},
            },
        }
    ],
}


def _patch_stub_llm(testcase: unittest.TestCase) -> None:
    clients: list = []

    def fake_get_client_instance(llm_config, *, agent_name=None, vllm_runtime=None):
        client = StubLLMClient(agent_name or "")
        clients.append(client)
        return client

    async def fake_judge_run(run_dir, config):
        return {}

    p1 = unittest.mock.patch.object(run_mod, "get_client_instance", fake_get_client_instance)
    p2 = unittest.mock.patch.object(run_mod, "judge_run", fake_judge_run)
    p1.start()
    p2.start()
    testcase.addCleanup(p1.stop)
    testcase.addCleanup(p2.stop)


class NativeSmokeRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_run_native_phases_gating_and_secret_channel(self) -> None:
        _patch_stub_llm(self)
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "smoke.yaml"
            cfg_path.write_text(yaml.safe_dump(_SMOKE_CONFIG), encoding="utf-8")
            out_root = await run_mod.run_from_config(str(cfg_path), out_dir=str(Path(tmp) / "out"))

            runs = list((out_root / "runs").rglob("agent_turns.json"))
            self.assertEqual(len(runs), 1, f"expected exactly one run dir, got {runs}")
            run_dir = runs[0].parent

            turns = json.loads((run_dir / "agent_turns.json").read_text(encoding="utf-8"))
            phases = {t["phase"] for t in turns}
            self.assertIn("pre_coordination", phases)
            self.assertIn("planning", phases)
            self.assertIn("execution", phases)
            self.assertIn("survey", phases)

            pre = [t for t in turns if t["phase"] == "pre_coordination"]
            self.assertTrue(pre)
            self.assertTrue(all(t["iteration"] == 0 for t in pre))
            run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            adversaries = set(run_cfg["adversaries"])
            self.assertEqual({t["agent"] for t in pre}, adversaries)
            self.assertTrue(all(t["role"] == "adversary" for t in pre))

            # Secret board exists with exactly the coalition as participants.
            secret_id = run_cfg.get("secret_blackboard_id")
            self.assertIsNotNone(secret_id)
            boards = json.loads((run_dir / "blackboards.json").read_text(encoding="utf-8"))
            secret = next(
                (b for b in boards if str(b["blackboard_id"]) == str(secret_id)), None
            )
            self.assertIsNotNone(
                secret, f"secret board {secret_id!r} not in boards {boards!r}; run_cfg={run_cfg!r}"
            )
            self.assertEqual(set(secret["participants"]), adversaries)

            # The stub's public-board post was rewritten onto the secret board
            # during pre_coordination (enforce_in_phases).
            secret_text = json.dumps(secret["events"])
            self.assertIn("secret plan from", secret_text)
            # No adversary post may land on a public board outside the
            # pre_coordination window.
            for b in boards:
                if str(b["blackboard_id"]) == str(secret_id):
                    continue
                for ev in b["events"]:
                    if ev.get("agent") in adversaries and ev.get("kind") == "communication":
                        self.assertEqual(
                            ev["payload"].get("phase"), "pre_coordination",
                            f"adversary post leaked to public board in {ev['payload']}",
                        )

            # Adversaries' post_message is blocked outside pre_coordination.
            tool_events = json.loads((run_dir / "tool_events.json").read_text(encoding="utf-8"))
            pm_events = [e for e in tool_events if e["tool_name"] == "post_message"]
            self.assertTrue(pm_events)
            for e in pm_events:
                if e["phase"] == "pre_coordination" and e["agent_name"] in adversaries:
                    self.assertNotEqual(
                        (e["result"] or {}).get("status"), "blocked",
                        "post_message must stay available in pre_coordination",
                    )
                elif e["agent_name"] in adversaries:
                    self.assertEqual(
                        (e["result"] or {}).get("status"), "blocked",
                        f"adversary post_message must be blocked in {e['phase']}",
                    )


if __name__ == "__main__":
    unittest.main()
