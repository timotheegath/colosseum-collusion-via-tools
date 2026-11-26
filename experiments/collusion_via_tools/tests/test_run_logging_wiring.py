"""Regression tests for run logging wiring in run.py.

- `_build_agent` must pass the run's tool logger through to the agent (the
  5th positional arg of `BaseAgent.__init__`). Passing `None` there silently
  drops every tool call, leaving the run dir's `tool_calls.json` empty.
- `_apply_experiment_tag` must derive `simulation.tags` from `experiment.tag`
  so legacy `logs/<env>/<tag_model>/...` dirs are named after the experiment
  instead of `unknown_<model>`.

Pure stdlib ``unittest``: no network, no LLM.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.collusion_via_tools.agents import AdversarySpec
from experiments.collusion_via_tools.prompt_capture import ThreadSafePromptRecorder
from experiments.collusion_via_tools.run import _apply_experiment_tag, _build_agent


class _FakeClient:
    """Stand-in for an LLM client; BaseAgent only stores it."""


class BuildAgentTests(unittest.TestCase):
    def test_tool_logger_is_wired_to_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ThreadSafePromptRecorder(Path(tmp))
            tool_logger = object()  # sentinel: identity is what matters
            agent = _build_agent(
                client=_FakeClient(),
                name="Triage_0",
                model_name="fake-model",
                max_steps=2,
                tool_logger=tool_logger,
                trajectory_logger=None,
                environment_name="HospitalEnvironment",
                generation_params={},
                adversary_spec=AdversarySpec(adversary_names=set()),
                prompt_recorder=recorder,
            )
        self.assertIs(agent.tool_logger, tool_logger)


class ApplyExperimentTagTests(unittest.TestCase):
    def test_derives_tags_from_experiment_tag(self) -> None:
        cfg = {"experiment": {"tag": "collusion_via_tools_1"}}
        _apply_experiment_tag(cfg)
        self.assertEqual(cfg["simulation"]["tags"], ["collusion_via_tools_1"])

    def test_existing_simulation_tags_win(self) -> None:
        cfg = {
            "experiment": {"tag": "collusion_via_tools_1"},
            "simulation": {"tags": ["custom"]},
        }
        _apply_experiment_tag(cfg)
        self.assertEqual(cfg["simulation"]["tags"], ["custom"])

    def test_no_experiment_tag_is_a_noop(self) -> None:
        cfg = {"experiment": {}, "simulation": {"seed": 1}}
        _apply_experiment_tag(cfg)
        self.assertNotIn("tags", cfg["simulation"])

    def test_missing_experiment_block_is_a_noop(self) -> None:
        cfg: dict = {}
        _apply_experiment_tag(cfg)
        self.assertNotIn("tags", cfg["simulation"])


if __name__ == "__main__":
    unittest.main()
