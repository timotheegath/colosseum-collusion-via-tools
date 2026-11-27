"""Tests that ``reasoning_effort`` set in the experiment config reaches the agents.

The agent side of the chain is ``run_from_config`` -> ``cfg["llm"] =
model["llm"]`` -> ``get_generation_params(cfg["llm"])`` -> agent generation
params -> client request params. These tests pin the config-loading half of
that chain against the shipped ``collusion_via_tools_1.yaml`` so a config
regression (e.g. the key being dropped) fails loudly.

No network or LLM call is involved.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict

import yaml

from terrarium.utils import get_generation_params

from experiments.collusion_via_tools.judge_beliefs import (
    JudgeConfig,
    _build_judge_params,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "collusion_via_tools_1.yaml"
)


def _load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class AgentReasoningEffortConfigTests(unittest.TestCase):
    def test_every_llm_model_entry_passes_reasoning_effort_to_agents(self) -> None:
        cfg = _load_config()
        models = cfg.get("llm_models") or []
        self.assertTrue(models, "expected at least one llm_models entry")
        for model in models:
            with self.subTest(label=model.get("label")):
                llm_cfg = model["llm"]
                generation_params = get_generation_params(llm_cfg)
                self.assertEqual(generation_params.get("reasoning_effort"), "high")

    def test_judge_llm_params_carry_reasoning_effort_into_judge_params(self) -> None:
        cfg = _load_config()
        judge_llm = cfg.get("judge_llm") or {}
        provider = judge_llm.get("provider") or "openai"
        provider_cfg = judge_llm.get(provider) or {}
        params = provider_cfg.get("params") or {}
        judge_cfg = JudgeConfig(
            model=provider_cfg.get("model") or "gpt-4",
            max_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", 0.0),
            reasoning_effort=params.get("reasoning_effort"),
        )
        built = _build_judge_params(judge_cfg)
        self.assertEqual(built.get("reasoning_effort"), "high")


if __name__ == "__main__":
    unittest.main()
