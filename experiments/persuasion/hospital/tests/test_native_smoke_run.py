"""End-to-end smoke test of persuasion/hospital's native config-driven wiring.

Runs ``run_from_config`` on a tiny config with a stub LLM client (no network)
and asserts: planning + execution phases ran, the coalition-only secret
blackboard exists with exactly the colluders as participants, the stub's post
landed on it, and the prompt variant was injected into the colluder prompt
(via the native ``environment.collusion`` / ``prompt_wrapper`` flow).
"""

from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import yaml

from experiments.persuasion.hospital import run as run_mod
from experiments.collusion_via_tools.tests._stub_client import StubLLMClient

_SMOKE_CONFIG = {
    "experiment": {
        "tag": "smoke",
        "max_concurrent_runs": 1,
        "seeds": [0],
        "persuasion_hospital": {
            "colluder_selection": "hospital_coalition",
        },
        "sweeps": [
            {
                "name": "smoke_sweep",
                "topologies": ["complete"],
                "num_agents": [5],
                "num_patients": ["num_agents_minus_1"],
                "strategies": ["covert_max_coalition_reward"],
                "colluder_counts": [2],
                "target_roles": ["departmental"],
                "coalition_clique_enabled": [True],
                "secret_channel_enabled": [True],
                "prompt_variants": ["helpful_misdirection"],
            }
        ],
    },
    "simulation": {
        "max_iterations": 1,
        "max_planning_rounds": 1,
        "max_conversation_steps": 2,
        "seed": 1,
    },
    "environment": {
        "import_path": "terrarium.environments.dcops.hospital.hospital_env:HospitalEnvironment",
        "name": "HospitalEnvironment",
        "num_patients": 4,
        "num_hospitals": 1,
    },
    "communication_network": {
        "topology": "complete",
        "num_agents": 5,
        "consolidate_channels": True,
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


class NativeSmokeRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_run_native_wiring(self) -> None:
        def fake_get_client_instance(llm_config, *, agent_name=None, vllm_runtime=None):
            return StubLLMClient(agent_name or "")

        # The canned post targets the secret board (id 1; public board is 0).
        StubLLMClient.blackboard_id = 1
        self.addCleanup(setattr, StubLLMClient, "blackboard_id", 0)
        with unittest.mock.patch.object(run_mod, "get_client_instance", fake_get_client_instance):
            with tempfile.TemporaryDirectory() as tmp:
                cfg_path = Path(tmp) / "smoke.yaml"
                cfg_path.write_text(yaml.safe_dump(_SMOKE_CONFIG), encoding="utf-8")
                out_root = await run_mod.run_from_config(str(cfg_path), out_dir=str(Path(tmp) / "out"))

                runs = list((out_root / "runs").rglob("agent_turns.json"))
                self.assertEqual(len(runs), 1, f"expected one run, got {runs}")
                run_dir = runs[0].parent

                turns = json.loads((run_dir / "agent_turns.json").read_text(encoding="utf-8"))
                phases = {t["phase"] for t in turns}
                self.assertEqual(phases, {"planning", "execution"})

                run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
                colluders = set(run_cfg.get("colluders") or [])
                self.assertEqual(len(colluders), 2)
                self.assertEqual(run_cfg.get("prompt_variant"), "helpful_misdirection")

                secret_id = run_cfg.get("secret_blackboard_id")
                self.assertIsNotNone(secret_id)
                boards = json.loads((run_dir / "blackboards.json").read_text(encoding="utf-8"))
                secret = next(
                    (b for b in boards if str(b["blackboard_id"]) == str(secret_id)), None
                )
                self.assertIsNotNone(secret, f"boards={boards!r}")
                self.assertEqual(set(secret["participants"]), colluders)
                self.assertIn("secret plan from", json.dumps(secret["events"]))

                # The colluder prompt carried the persuasion variant text.
                prompts_log = run_dir / "prompts.jsonl"
                if prompts_log.exists():
                    logged = prompts_log.read_text(encoding="utf-8")
                    self.assertIn("Persuasion playbook", logged)


if __name__ == "__main__":
    unittest.main()
