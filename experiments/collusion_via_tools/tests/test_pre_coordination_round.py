"""Pre-coordination phase tests (native config-driven wiring).

The phase itself is now expressed with native submodule config
(``simulation.phases`` entry with ``when: "before"`` + ``agents`` subset +
``max_calls``, ``environment.blocked_tools`` and ``environment.secret_channels``)
and walked by ``run_phase_entries`` through the generic ``protocol.agent_turn``.
This file keeps the resolver/run-id tests for the sweep axes and adds tests for
the native gating/rewrite/cap behavior the runner now relies on.
"""

from __future__ import annotations

import asyncio
import types
import unittest

from experiments.common.run_utils import (
    inject_native_collusion_config,
    run_phase_entries,
)
from experiments.collusion_via_tools import run as run_mod
from experiments.collusion_via_tools.plots import build_tables as bt
from terrarium.tools.gates import (
    call_limit_response,
    secret_channel_rewrite,
    tool_blocked,
)
from terrarium.utils import phase_max_calls, resolve_phases, split_phases_by_when


class NormalizePreCoordinationValueTests(unittest.TestCase):
    def test_truthy_values(self) -> None:
        for v in (True, 1, "1", "true", "yes", "on", "TRUE"):
            with self.subTest(v=v):
                self.assertTrue(run_mod._normalize_pre_coordination_value(v))

    def test_falsy_values(self) -> None:
        for v in (False, 0, None, "", "no", "false"):
            with self.subTest(v=v):
                self.assertFalse(run_mod._normalize_pre_coordination_value(v))


class ResolvePreCoordinationRoundTests(unittest.TestCase):
    def test_explicit_value_wins(self) -> None:
        # Even with a top-level config setting it to True, an explicit False
        # at the call site is honoured.
        cfg = {"experiment": {"pre_coordination_round": True}}
        self.assertFalse(run_mod._resolve_pre_coordination_round(False, cfg))

    def test_default_is_false(self) -> None:
        self.assertFalse(run_mod._resolve_pre_coordination_round(run_mod._UNSET, {}))
        self.assertFalse(run_mod._resolve_pre_coordination_round(run_mod._UNSET, None))

    def test_top_level_resolution_order(self) -> None:
        cfg = {
            "experiment": {"pre_coordination_round": True},
            "communication_network": {"pre_coordination_round": False},
        }
        self.assertTrue(run_mod._resolve_pre_coordination_round(run_mod._UNSET, cfg))

        # `adversary` lives under `experiment`, matching the disable_pm resolution.
        cfg2 = {"experiment": {"adversary": {"pre_coordination_round": True}}}
        self.assertTrue(run_mod._resolve_pre_coordination_round(run_mod._UNSET, cfg2))

        cfg3 = {
            "experiment": {},
            "communication_network": {"pre_coordination_round": True},
        }
        self.assertTrue(run_mod._resolve_pre_coordination_round(run_mod._UNSET, cfg3))


class ResolvePreCoordinationAxisTests(unittest.TestCase):
    def test_sweep_value_wins_over_top_level(self) -> None:
        cfg = {"experiment": {"pre_coordination_round": True}}
        # Sweep-level scalar False overrides top-level True.
        self.assertEqual(
            run_mod._resolve_pre_coordination_axis(
                {"pre_coordination_round": False}, cfg
            ),
            [False],
        )

    def test_sweep_list_expands_into_one_decision_per_run(self) -> None:
        cfg = {}
        self.assertEqual(
            run_mod._resolve_pre_coordination_axis(
                {"pre_coordination_round": [True, False]}, cfg
            ),
            [True, False],
        )

    def test_top_level_falls_through_when_sweep_key_absent(self) -> None:
        cfg = {"experiment": {"pre_coordination_round": True}}
        self.assertEqual(
            run_mod._resolve_pre_coordination_axis({}, cfg),
            [True],
        )

    def test_default_is_off(self) -> None:
        self.assertEqual(run_mod._resolve_pre_coordination_axis({}, {}), [False])


class BuildRunIdSuffixTests(unittest.TestCase):
    BASE_KW = dict(
        model_label="m",
        sweep_name="s",
        topology="complete",
        num_agents=9,
        num_patients=8,
        adversary_count=2,
        target_role="Resource_Provisioner",
        seed=1,
    )

    def test_no_suffix_when_disabled(self) -> None:
        rid = run_mod._build_run_id(**self.BASE_KW)
        self.assertNotIn("__precoord", rid)
        self.assertFalse(rid.endswith("__nopm"))

    def test_precoord_suffix_is_always_counted(self) -> None:
        # The count is a sweep axis, so it is ALWAYS encoded -- even for 1.
        # The legacy count-less `__precoord` must never be produced.
        for rounds, expected in (
            (1, "__precoord1"),
            (2, "__precoord2"),
            (3, "__precoord3"),
        ):
            with self.subTest(rounds=rounds):
                rid = run_mod._build_run_id(
                    **self.BASE_KW,
                    pre_coordination_round=True,
                    pre_coordination_rounds=rounds,
                )
                self.assertTrue(rid.endswith(expected))
                self.assertFalse(rid.endswith("__precoord"))

    def test_precoord_with_nopm(self) -> None:
        rid = run_mod._build_run_id(
            **self.BASE_KW,
            disable_post_message=True,
            pre_coordination_round=True,
            pre_coordination_rounds=2,
        )
        self.assertTrue(rid.endswith("__nopm__precoord2"))

    def test_nopm_alone(self) -> None:
        rid = run_mod._build_run_id(**self.BASE_KW, disable_post_message=True)
        self.assertTrue(rid.endswith("__nopm"))
        self.assertNotIn("__precoord", rid)

    def test_different_counts_produce_different_ids(self) -> None:
        rid_one = run_mod._build_run_id(
            **self.BASE_KW,
            pre_coordination_round=True,
            pre_coordination_rounds=1,
        )
        rid_two = run_mod._build_run_id(
            **self.BASE_KW,
            pre_coordination_round=True,
            pre_coordination_rounds=2,
        )
        self.assertNotEqual(rid_one, rid_two)

    def test_precoord_false_does_not_change_id(self) -> None:
        rid_a = run_mod._build_run_id(**self.BASE_KW, disable_post_message=True)
        # The count is irrelevant when the phase is off: no suffix at all.
        rid_b = run_mod._build_run_id(
            **self.BASE_KW,
            disable_post_message=True,
            pre_coordination_round=False,
            pre_coordination_rounds=3,
        )
        self.assertEqual(rid_a, rid_b)
        self.assertNotIn("__precoord", rid_b)


class BuildTablesRegexTests(unittest.TestCase):
    def test_precoord_alone(self) -> None:
        m = bt._RUN_ID_RE_V2.match(
            "m__s__topocomplete__agents9_patients8_adv2_Resource_Provisioner_seed1__precoord"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group("precoord"), "__precoord")
        self.assertIsNone(m.group("precoord_rounds"))
        self.assertIsNone(m.group("restriction"))

    def test_precoord_counted(self) -> None:
        for n in (1, 2, 10):
            with self.subTest(n=n):
                m = bt._RUN_ID_RE_V2.match(
                    f"m__s__topocomplete__agents9_patients8_adv2_Resource_Provisioner_seed1__precoord{n}"
                )
                self.assertIsNotNone(m)
                self.assertEqual(m.group("precoord"), f"__precoord{n}")
                self.assertEqual(m.group("precoord_rounds"), str(n))

    def test_nopm_with_precoord(self) -> None:
        m = bt._RUN_ID_RE_V2.match(
            "m__s__topocomplete__agents9_patients8_adv2_Resource_Provisioner_seed1__nopm__precoord"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group("restriction"), "nopm")
        self.assertEqual(m.group("precoord"), "__precoord")

    def test_nopm_with_counted_precoord(self) -> None:
        m = bt._RUN_ID_RE_V2.match(
            "m__s__topocomplete__agents9_patients8_adv2_Resource_Provisioner_seed1__nopm__precoord2"
        )
        self.assertIsNotNone(m)
        self.assertEqual(m.group("restriction"), "nopm")
        self.assertEqual(m.group("precoord"), "__precoord2")
        self.assertEqual(m.group("precoord_rounds"), "2")

    def test_legacy_cut_after_r_suffix_no_longer_parses(self) -> None:
        # The `disable_post_message_after_round: N` axis is gone, so old run
        # ids intentionally stop parsing (they fall back to the directory
        # heuristic in `_infer_run_fields`).
        m = bt._RUN_ID_RE_V2.match(
            "m__s__topocomplete__agents9_patients8_adv2_Resource_Provisioner_seed1__cut_after_r1__precoord"
        )
        self.assertIsNone(m)

    def test_no_suffix_round_trip(self) -> None:
        m = bt._RUN_ID_RE_V2.match(
            "m__s__topocomplete__agents9_patients8_adv2_Resource_Provisioner_seed1"
        )
        self.assertIsNotNone(m)
        self.assertIsNone(m.group("restriction"))
        self.assertIsNone(m.group("precoord"))



class NormalizePreCoordinationMaxMessagesTests(unittest.TestCase):
    def test_none_is_unlimited(self) -> None:
        for v in (None, "null", "none", "unlimited", ""):
            with self.subTest(v=v):
                self.assertIsNone(run_mod._normalize_pre_coordination_max_messages(v))

    def test_positive_int_is_clamped(self) -> None:
        for v, expected in [(1, 1), (3, 3), (100, 100), ("5", 5)]:
            with self.subTest(v=v):
                self.assertEqual(
                    run_mod._normalize_pre_coordination_max_messages(v), expected
                )

    def test_zero_is_unlimited(self) -> None:
        # 0 is meaningless as a cap and is treated as "no cap" so existing
        # configs that accidentally set 0 don't suddenly start rejecting all
        # messages.
        for v in (0, 0.0, "0"):
            with self.subTest(v=v):
                self.assertIsNone(run_mod._normalize_pre_coordination_max_messages(v))

    def test_bool_meaning(self) -> None:
        # True -> 1 message, False -> unlimited. Avoids accidentally
        # interpreting `True` as a giant integer.
        self.assertEqual(run_mod._normalize_pre_coordination_max_messages(True), 1)
        self.assertIsNone(run_mod._normalize_pre_coordination_max_messages(False))

    def test_garbage_falls_back_to_unlimited(self) -> None:
        self.assertIsNone(run_mod._normalize_pre_coordination_max_messages("abc"))
        self.assertIsNone(run_mod._normalize_pre_coordination_max_messages({}))


class ResolvePreCoordinationMaxMessagesTests(unittest.TestCase):
    def test_default_is_unlimited(self) -> None:
        self.assertIsNone(run_mod._resolve_pre_coordination_max_messages({}, {}))

    def test_sweep_value_wins(self) -> None:
        sweep = {"pre_coordination_max_messages": 3}
        cfg = {"experiment": {"pre_coordination_max_messages": 99}}
        self.assertEqual(run_mod._resolve_pre_coordination_max_messages(sweep, cfg), 3)

    def test_top_level_resolution(self) -> None:
        cfg = {"experiment": {"pre_coordination_max_messages": 5}}
        self.assertEqual(run_mod._resolve_pre_coordination_max_messages({}, cfg), 5)

        cfg2 = {
            "experiment": {
                "adversary": {"pre_coordination_max_messages": 7},
            }
        }
        self.assertEqual(run_mod._resolve_pre_coordination_max_messages({}, cfg2), 7)

        cfg3 = {
            "experiment": {},
            "communication_network": {"pre_coordination_max_messages": 9},
        }
        self.assertEqual(run_mod._resolve_pre_coordination_max_messages({}, cfg3), 9)

    def test_scalar_only(self) -> None:
        # The max-messages knob must not introduce its own sweep dimension;
        # a list is rejected (treated as "no cap") rather than silently taking
        # the first element.
        result = run_mod._resolve_pre_coordination_max_messages(
            {"pre_coordination_max_messages": [1, 2, 3]}, {}
        )
        self.assertIsNone(result)



class EffectiveSecretChannelPreCoordTests(unittest.TestCase):
    """``disable_post_message: true`` must still create the secret blackboard
    when the pre-coordination round is enabled -- otherwise the adversary
    leaks the coordination scheme to the public board.
    """

    def test_disabled_with_precoord_creates_secret_board(self) -> None:
        self.assertTrue(
            run_mod._compute_effective_secret_channel(
                secret_channel_enabled=True,
                disable_post_message=True,
                pre_coordination_round=True,
            )
        )

    def test_disabled_without_precoord_suppresses_secret_board(self) -> None:
        self.assertFalse(
            run_mod._compute_effective_secret_channel(
                secret_channel_enabled=True,
                disable_post_message=True,
                pre_coordination_round=False,
            )
        )

    def test_enabled_always_creates_secret_board(self) -> None:
        for precoord in (True, False):
            with self.subTest(precoord=precoord):
                self.assertTrue(
                    run_mod._compute_effective_secret_channel(
                        secret_channel_enabled=True,
                        disable_post_message=False,
                        pre_coordination_round=precoord,
                    )
                )

    def test_disabled_request_still_suppresses(self) -> None:
        self.assertFalse(
            run_mod._compute_effective_secret_channel(
                secret_channel_enabled=False,
                disable_post_message=False,
                pre_coordination_round=True,
            )
        )


class EffectiveSecretChannelActualConditionsTests(unittest.TestCase):
    """The recorded flag must match the conditions that actually create the
    board: non-benign, at least two adversaries, and the tooling rules.
    """

    def _resolve(self, **overrides):
        kwargs = dict(
            secret_channel_enabled=True,
            disable_post_message=True,
            pre_coordination_round=True,
            is_benign=False,
            num_adversaries=2,
        )
        kwargs.update(overrides)
        return run_mod._resolve_effective_secret_channel(**kwargs)

    def test_one_adversary_precoord_is_not_effective(self) -> None:
        # The phase may run (with no private channel) but no board is created.
        self.assertFalse(self._resolve(num_adversaries=1))

    def test_two_adversaries_precoord_is_effective(self) -> None:
        self.assertTrue(self._resolve(num_adversaries=2))

    def test_benign_run_is_never_effective(self) -> None:
        self.assertFalse(self._resolve(is_benign=True, num_adversaries=0))

    def test_tooling_rules_still_apply(self) -> None:
        self.assertFalse(self._resolve(pre_coordination_round=False))
        self.assertFalse(self._resolve(secret_channel_enabled=False))



class PreCoordinationRoundsResolutionTests(unittest.TestCase):
    """`pre_coordination_rounds` is strict and backward compatible.

    The global `simulation.pre_coordination_rounds` is the fallback; the
    per-sweep axis is covered by `ResolvePreCoordinationRoundsAxisTests`.
    """

    def test_default_is_one(self) -> None:
        self.assertEqual(run_mod._resolve_pre_coordination_rounds({}), 1)
        self.assertEqual(
            run_mod._resolve_pre_coordination_rounds({"simulation": {}}), 1
        )
        self.assertEqual(run_mod._resolve_pre_coordination_rounds(None), 1)

    def test_configured_value_is_used(self) -> None:
        cfg = {"simulation": {"pre_coordination_rounds": 2}}
        self.assertEqual(run_mod._resolve_pre_coordination_rounds(cfg), 2)

        cfg3 = {"simulation": {"pre_coordination_rounds": 4}}
        self.assertEqual(run_mod._resolve_pre_coordination_rounds(cfg3), 4)

    def test_shipped_config_no_global_count_and_sweep_declares_axis(self) -> None:
        # The count moved to the sweep axis: the shipped config no longer sets
        # the global value (so the fallback resolves to 1), and the sweeps that
        # declare the axis override it.
        import yaml
        from pathlib import Path

        cfg_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "collusion_via_tools_1.yaml"
        )
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(run_mod._resolve_pre_coordination_rounds(cfg), 1)
        self.assertNotIn("pre_coordination_rounds", cfg.get("simulation") or {})

        sweeps = (cfg.get("experiment") or {}).get("sweeps") or []
        by_name = {s.get("name"): s for s in sweeps}
        benign = by_name["benign_baseline_sweep"]
        self.assertEqual(
            run_mod._resolve_pre_coordination_rounds_axis(
                benign, cfg, phase_enabled=False
            ),
            [1],
        )

    def test_invalid_values_are_rejected(self) -> None:
        # Booleans are `int` subclasses, so they need an explicit rejection.
        for bad in (
            True,
            False,
            0,
            -1,
            1.5,
            2.0,
            "2",
            None,
            [],
            {},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run_mod._normalize_pre_coordination_rounds(bad)

    def test_error_message_is_clear(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            run_mod._normalize_pre_coordination_rounds(0)
        self.assertIn("pre_coordination_rounds", str(ctx.exception))
        self.assertIn(">= 1", str(ctx.exception))


class ResolvePreCoordinationRoundsAxisTests(unittest.TestCase):
    """The `sweeps[].pre_coordination_rounds` sweep axis."""

    def test_absent_sweep_key_falls_through_to_global(self) -> None:
        cfg = {"simulation": {"pre_coordination_rounds": 3}}
        self.assertEqual(run_mod._resolve_pre_coordination_rounds_axis({}, cfg), [3])

    def test_global_default_is_one(self) -> None:
        self.assertEqual(run_mod._resolve_pre_coordination_rounds_axis({}, {}), [1])
        self.assertEqual(run_mod._resolve_pre_coordination_rounds_axis({}, None), [1])

    def test_sweep_scalar_wins_over_global(self) -> None:
        cfg = {"simulation": {"pre_coordination_rounds": 3}}
        self.assertEqual(
            run_mod._resolve_pre_coordination_rounds_axis(
                {"pre_coordination_rounds": 2}, cfg
            ),
            [2],
        )

    def test_sweep_list_expands_to_one_value_per_run(self) -> None:
        cfg = {"simulation": {"pre_coordination_rounds": 3}}
        self.assertEqual(
            run_mod._resolve_pre_coordination_rounds_axis(
                {"pre_coordination_rounds": [1, 2]}, cfg
            ),
            [1, 2],
        )
        self.assertEqual(
            run_mod._resolve_pre_coordination_rounds_axis(
                {"pre_coordination_rounds": [2, 1, 4]}, cfg
            ),
            [2, 1, 4],
        )

    def test_phase_off_collapses_to_single_value(self) -> None:
        # The count is inert when the phase is off; expanding it would make two
        # runs share a run_id (a directory collision).
        cfg = {}
        self.assertEqual(
            run_mod._resolve_pre_coordination_rounds_axis(
                {"pre_coordination_rounds": [1, 2]}, cfg, phase_enabled=False
            ),
            [1],
        )
        self.assertEqual(
            run_mod._resolve_pre_coordination_rounds_axis(
                {"pre_coordination_rounds": [3, 4]}, cfg, phase_enabled=False
            ),
            [3],
        )

    def test_phase_on_keeps_all_values(self) -> None:
        self.assertEqual(
            run_mod._resolve_pre_coordination_rounds_axis(
                {"pre_coordination_rounds": [1, 2]}, {}, phase_enabled=True
            ),
            [1, 2],
        )

    def test_invalid_values_are_rejected(self) -> None:
        for bad in (True, False, 0, -1, 1.5, "2", None, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run_mod._resolve_pre_coordination_rounds_axis(
                        {"pre_coordination_rounds": bad}, {}
                    )

    def test_invalid_element_in_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_mod._resolve_pre_coordination_rounds_axis(
                {"pre_coordination_rounds": [1, 0]}, {}
            )



class TrajectoryRerunResetTests(unittest.TestCase):
    """A resume/rerun reuses the run directory; stale pre-coordination rounds
    must not survive into the new trajectory file.
    """

    def test_reset_clears_stale_pre_coordination_rounds(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from terrarium.core.logger import AgentTrajectoryLogger

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            logger = AgentTrajectoryLogger(
                environment_name="HospitalEnvironment",
                seed=1,
                config={"simulation": {"run_timestamp": "stale"}},
                log_dir=run_dir,
            )
            logger.log_trajectory(
                agent_name="Adv1",
                iteration=0,
                phase="pre_coordination",
                trajectory_dict={"step_1": {"reasoning": "stale round"}},
                round_num=1,
            )
            stale = json.loads(
                (run_dir / "agent_trajectories.json").read_text(encoding="utf-8")
            )
            self.assertIn("round_1", stale["Adv1"]["iteration_0"]["pre_coordination"])

            # `run._run_single` performs this reset before any turn is logged.
            logger.reset_log()
            fresh = json.loads(
                (run_dir / "agent_trajectories.json").read_text(encoding="utf-8")
            )
            self.assertEqual(fresh, {})


if __name__ == "__main__":
    unittest.main()


def _gate_env(blocked_tools=None, secret_channels=None, secret_channel_by_agent=None, role_by_agent=None):
    """Minimal env stub exposing the attributes the native gates read."""
    return types.SimpleNamespace(
        env_config={
            "blocked_tools": blocked_tools or {},
            "secret_channels": secret_channels or [],
        },
        secret_channel_by_agent=secret_channel_by_agent or {},
        role_by_agent=role_by_agent or {},
    )


class NativeBlockedToolsGateTests(unittest.TestCase):
    """``environment.blocked_tools`` is the native replacement for the removed
    ContextVar gating: wildcard blocks env tools but never blackboard tools."""

    def test_wildcard_blocks_env_tools_not_post_message(self) -> None:
        env = _gate_env(blocked_tools={"pre_coordination": ["*"]})
        self.assertTrue(tool_blocked(env, "broadcast_message", "pre_coordination", "AgentA"))
        self.assertTrue(tool_blocked(env, "transfer_resources", "pre_coordination", "AgentA"))
        self.assertFalse(tool_blocked(env, "post_message", "pre_coordination", "AgentA"))

    def test_wildcard_plus_post_message_blocks_both(self) -> None:
        env = _gate_env(blocked_tools={"pre_coordination": ["*", "post_message"]})
        self.assertTrue(tool_blocked(env, "post_message", "pre_coordination", "AgentA"))
        self.assertTrue(tool_blocked(env, "schedule_patient", "pre_coordination", "AgentA"))

    def test_roles_scoped_entry_applies_to_adversaries_only(self) -> None:
        env = _gate_env(
            blocked_tools={
                "planning": [{"tool": "post_message", "roles": ["adversary"]}],
            },
            role_by_agent={"AgentA": "adversary", "AgentB": "cooperative"},
        )
        self.assertTrue(tool_blocked(env, "post_message", "planning", "AgentA"))
        self.assertFalse(tool_blocked(env, "post_message", "planning", "AgentB"))
        self.assertFalse(tool_blocked(env, "post_message", "execution", "AgentA"))


class NativeSecretChannelRewriteTests(unittest.TestCase):
    """``environment.secret_channels[].enforce_in_phases`` drives the native
    rewrite of post_message onto the coalition-only board."""

    def _env(self):
        return _gate_env(
            secret_channels=[
                {"members": ["AgentA", "AgentB"], "enforce_in_phases": ["pre_coordination"]}
            ],
            secret_channel_by_agent={"AgentA": 7, "AgentB": 7, "AgentC": None},
        )

    def test_rewrite_applies_in_enforced_phase_for_members(self) -> None:
        env = self._env()
        self.assertEqual(secret_channel_rewrite(env, "AgentA", "pre_coordination"), 7)
        self.assertEqual(secret_channel_rewrite(env, "AgentB", "pre_coordination"), 7)

    def test_no_rewrite_outside_enforced_phase_or_for_non_members(self) -> None:
        env = self._env()
        self.assertIsNone(secret_channel_rewrite(env, "AgentA", "planning"))
        self.assertIsNone(secret_channel_rewrite(env, "AgentC", "pre_coordination"))


class NativeMaxCallsTests(unittest.TestCase):
    def test_phase_max_calls_read_from_phase_entry(self) -> None:
        phases = resolve_phases(
            {
                "phases": [
                    {
                        "name": "pre_coordination",
                        "rounds": 2,
                        "when": "before",
                        "agents": ["AgentA"],
                        "max_calls": {"post_message": 3},
                    }
                ]
            }
        )
        self.assertEqual(phase_max_calls(phases, "pre_coordination"), {"post_message": 3})
        self.assertEqual(phase_max_calls(phases, "planning"), {})
        groups = split_phases_by_when(phases)
        self.assertEqual([p["name"] for p in groups["before"]], ["pre_coordination"])

    def test_call_limit_response_shape(self) -> None:
        resp = call_limit_response("post_message", "pre_coordination", 3, 3)
        self.assertTrue(resp["blocked"])
        self.assertEqual(resp["status"], "call_limit")
        self.assertEqual(resp["limit"], 3)


class InjectNativeCollusionConfigTests(unittest.TestCase):
    def test_writes_native_blocks(self) -> None:
        cfg = {"simulation": {"max_iterations": 1}, "environment": {"name": "HospitalEnvironment"}}
        phases = [
            {
                "name": "pre_coordination",
                "rounds": 2,
                "when": "before",
                "agents": ["AgentA", "AgentB"],
                "max_calls": {"post_message": 4},
            },
            {"name": "planning", "rounds": 1},
            {"name": "execution", "rounds": 1},
            {"name": "survey", "rounds": 1, "when": "after"},
        ]
        inject_native_collusion_config(
            cfg,
            adversaries=["AgentA", "AgentB"],
            strategy="covert",
            instructions_file="collusion_instructions.txt",
            secret_channel=True,
            enforce_phase="pre_coordination",
            disable_post_message=True,
            prompt_wrapper="experiments.collusion_via_tools.prompts:MisalignmentPrompts",
            phases=phases,
            blocked_tools={"pre_coordination": ["*"]},
        )
        env_cfg = cfg["environment"]
        self.assertEqual(env_cfg["collusion"]["members"], ["AgentA", "AgentB"])
        self.assertEqual(env_cfg["collusion"]["instructions_file"], "collusion_instructions.txt")
        self.assertEqual(env_cfg["secret_channels"], [
            {"members": ["AgentA", "AgentB"], "enforce_in_phases": ["pre_coordination"]}
        ])
        self.assertEqual(env_cfg["blocked_tools"]["pre_coordination"], ["*"])
        pm_blocks = {p: env_cfg["blocked_tools"][p] for p in ("planning", "execution", "survey")}
        for entries in pm_blocks.values():
            self.assertIn({"tool": "post_message", "roles": ["adversary"]}, entries)
        self.assertEqual(env_cfg["prompt_wrapper"], "experiments.collusion_via_tools.prompts:MisalignmentPrompts")
        self.assertEqual(cfg["simulation"]["phases"], phases)


class _PhaseWalkAgent:
    def __init__(self, name: str) -> None:
        self.name = name


class _PhaseWalkEnv:
    def __init__(self, agent_names) -> None:
        self.agents = [_PhaseWalkAgent(n) for n in agent_names]
        self.context_calls = []

    def build_agent_context(self, agent_name, phase, iteration, **kwargs):
        self.context_calls.append(
            {"agent_name": agent_name, "phase": phase, "iteration": iteration, **kwargs}
        )
        return {"agent_name": agent_name, "phase": phase, "iteration": iteration}


class _PhaseWalkProtocol:
    def __init__(self, fail_on=None) -> None:
        self.calls = []
        self.fail_on = fail_on or set()

    async def agent_turn(self, agent, agent_name, agent_context, environment, iteration, phase, round_num=None):
        self.calls.append(
            {
                "agent_name": agent_name,
                "phase": phase,
                "iteration": iteration,
                "round_num": round_num,
            }
        )
        if agent_name in self.fail_on:
            raise RuntimeError("boom")
        return {"response": f"{agent_name}:{phase}:{round_num}", "usage": {}, "tools_executed": []}


class RunPhaseEntriesTests(unittest.IsolatedAsyncioTestCase):
    """The runner walks phases through the generic ``agent_turn``; the
    pre_coordination entry runs once with iteration=0, adversaries only."""

    def _build_turn(self, *, agent, phase, round_num, iteration, response, error=None):
        agent_name = agent if isinstance(agent, str) else agent.name
        return {
            "phase": phase,
            "iteration": iteration,
            "round": round_num,
            "agent": agent_name,
            "response": f"[ERROR] {error}" if error is not None else (response or {}).get("response"),
        }

    async def test_before_group_runs_adversaries_only_at_iteration_zero(self) -> None:
        env = _PhaseWalkEnv(["AgentA", "AgentB", "AgentC"])
        protocol = _PhaseWalkProtocol()
        phases = resolve_phases(
            {
                "phases": [
                    {"name": "pre_coordination", "rounds": 2, "when": "before", "agents": ["AgentA", "AgentB"]},
                    {"name": "planning", "rounds": 1},
                    {"name": "execution", "rounds": 1},
                    {"name": "survey", "rounds": 1, "when": "after"},
                ]
            }
        )
        groups = split_phases_by_when(phases)
        turns = await run_phase_entries(
            protocol=protocol,
            env=env,
            phase_entries=groups["before"],
            iteration=0,
            build_turn=self._build_turn,
        )
        self.assertEqual(
            [(c["agent_name"], c["phase"], c["iteration"], c["round_num"]) for c in protocol.calls],
            [
                ("AgentA", "pre_coordination", 0, 1),
                ("AgentB", "pre_coordination", 0, 1),
                ("AgentA", "pre_coordination", 0, 2),
                ("AgentB", "pre_coordination", 0, 2),
            ],
        )
        self.assertEqual(len(turns), 4)
        self.assertTrue(all(t["iteration"] == 0 for t in turns))

    async def test_each_and_after_groups_walk_all_agents(self) -> None:
        env = _PhaseWalkEnv(["AgentA", "AgentB"])
        protocol = _PhaseWalkProtocol()
        groups = split_phases_by_when(
            resolve_phases(
                {
                    "phases": [
                        {"name": "planning", "rounds": 1},
                        {"name": "execution", "rounds": 1},
                        {"name": "survey", "rounds": 1, "when": "after"},
                    ]
                }
            )
        )
        turns = await run_phase_entries(
            protocol=protocol, env=env, phase_entries=groups["each"],
            iteration=3, build_turn=self._build_turn,
        )
        self.assertEqual([(c["phase"], c["iteration"]) for c in protocol.calls],
                         [("planning", 3), ("planning", 3), ("execution", 3), ("execution", 3)])
        after = await run_phase_entries(
            protocol=protocol, env=env, phase_entries=groups["after"],
            iteration=3, build_turn=self._build_turn,
        )
        self.assertEqual([(c["phase"], c["iteration"]) for c in protocol.calls[-2:]],
                         [("survey", 3), ("survey", 3)])
        self.assertEqual(len(turns) + len(after), 6)

    async def test_turn_failure_is_recorded_not_raised(self) -> None:
        env = _PhaseWalkEnv(["AgentA", "AgentB"])
        protocol = _PhaseWalkProtocol(fail_on={"AgentA"})
        groups = split_phases_by_when(resolve_phases({"phases": [{"name": "planning", "rounds": 1}]}))
        turns = await run_phase_entries(
            protocol=protocol, env=env, phase_entries=groups["each"],
            iteration=1, build_turn=self._build_turn,
        )
        self.assertEqual(len(turns), 2)
        self.assertIn("[ERROR]", turns[0]["response"])
        self.assertNotIn("[ERROR]", turns[1]["response"])


if __name__ == "__main__":
    unittest.main()
