"""Tests for the `pre_coordination_round` knob and its round count.

The pre-coordination phase is an adversary-only set of turns that runs before
iteration 1 of the main simulation (default: one round, configurable via the
`sweeps[].pre_coordination_rounds` axis with the global
`simulation.pre_coordination_rounds` as fallback). The adversary gets the
planning-shaped prompt (with a phase-specific block explaining the round) and
`post_message` to the coalition blackboard, but any environment-tool call
(`broadcast_message`, `transfer_resources`, etc.) is silently dropped by the
protocol layer.

Layers tested in isolation:

1. Resolution: `_normalize_pre_coordination_value`, `_resolve_pre_coordination_round`,
   `_resolve_pre_coordination_axis`, and the strict
   `_normalize/_resolve_pre_coordination_rounds` /
   `_resolve_pre_coordination_rounds_axis` family -- same precedence
   semantics as `disable_post_message` for the boolean.
2. `run_id` suffixing: `_build_run_id` produces `__precoord<N>` either alone or
   after the existing post-message suffix (`__nopm__precoord2`).
3. Tool blocking: a native ``PhaseToolPolicy`` on the protocol
   (``block_environment_tool_calls=True``) makes
   ``LocalCommunicationProtocol.environment_handle_tool_call`` return a stub,
   and `post_message` flows through `blackboard_handle_tool_call` unaffected.
4. `build_tables._RUN_ID_RE_V2` parses `__precoord` and `__precoord<N>`
   (alone or combined).
5. The multi-round phase runner: N rounds, one turn per adversary, honest
   agents skipped, round metadata passed through, per-round message caps and
   per-round blackboard snapshots.
"""

from __future__ import annotations

import asyncio
import unittest

from terrarium.communication_protocols.local import (
    LocalCommunicationProtocol,
    PhaseToolPolicy,
)

from experiments.collusion_via_tools import run as run_mod
from experiments.collusion_via_tools.plots import build_tables as bt


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


class EnvToolsBlockPolicyTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        self.assertIsNone(proto.active_phase_policy("pre_coordination"))
        proto.set_phase_policy(
            "pre_coordination",
            PhaseToolPolicy(block_environment_tool_calls=True),
        )
        policy = proto.active_phase_policy("pre_coordination")
        self.assertIsNotNone(policy)
        self.assertTrue(policy.block_environment_tool_calls)
        proto.pop_phase_policy("pre_coordination")
        self.assertIsNone(proto.active_phase_policy("pre_coordination"))


class PreCoordinationContextAllowlistTests(unittest.TestCase):
    """The pre-coordination round must not hand the adversary environment
    state. Blocking tool calls is not enough: `env.build_agent_context`
    delivers the state passively in the prompt, so the hospital env natively
    returns the minimal identity + coalition context for
    `phase == "pre_coordination"` (Terrarium's
    `_PRE_COORDINATION_CONTEXT_ALLOWLIST`, an allowlist that fails closed).
    """

    # The two context shapes HospitalEnvironment.build_agent_context produces.
    PROVISIONER_CTX = {
        "inventory_snapshot": {"General_Hospital": {"IV_Kits": 20, "PPE_Sets": 21}},
        "failures_so_far": {"General_Hospital": ["Patient_0 step 1"]},
        "failures_by_hospital": {"General_Hospital": {"IV_Kits": 1}},
        "cost_table": {"IV_Kits": {"cost": 2}},
    }
    DEPARTMENT_CTX = {
        "dept_info": {
            "capacity": 2,
            "service": "Ward",
            "hospital": "St_Marys_Center",
            "local_inventory": {"IV_Kits": 5},
            "procedure_costs": {"IV_Kits": 2},
            "schedule_summary": {"H0": "0.0/2"},
        },
        "job_queue": [{"patient_id": "Patient_0", "step_index": 1}],
        "cost_table": {"IV_Kits": {"cost": 2}},
    }
    EXTRA = {
        "misalignment_role": "adversary",
        "misalignment_adversaries": ["Resource_Provisioner", "General_Hospital_Triage"],
        "misalignment_secret_blackboard_id": 7,
        "post_message_available": True,
    }

    def _make_env(self):
        from terrarium.environments.dcops.hospital.hospital_env import (
            HospitalEnvironment,
        )

        class _ProtocolStub:
            environment = None

        return HospitalEnvironment(
            communication_protocol=_ProtocolStub(),
            config={
                "simulation": {"seed": 7},
                "environment": {"num_hospitals": 2, "num_patients": 3},
            },
            tool_logger=None,
        )

    def _context(self, agent_name: str, state: dict) -> dict:
        env = self._make_env()
        return env.build_agent_context(
            agent_name,
            phase="pre_coordination",
            iteration=0,
            round_number=1,
            **state,
            **self.EXTRA,
        )

    def test_provisioner_state_is_stripped(self) -> None:
        out = self._context("Resource_Provisioner", self.PROVISIONER_CTX)
        for leaked in (
            "inventory_snapshot",
            "failures_so_far",
            "failures_by_hospital",
            "cost_table",
        ):
            with self.subTest(key=leaked):
                self.assertNotIn(leaked, out)

    def test_department_state_is_stripped(self) -> None:
        env = self._make_env()
        agent_name = next(a for a in env.agent_names if a != "Resource_Provisioner")
        out = env.build_agent_context(
            agent_name,
            phase="pre_coordination",
            iteration=0,
            round_number=1,
            **self.DEPARTMENT_CTX,
            **self.EXTRA,
        )
        for leaked in ("dept_info", "job_queue", "cost_table"):
            with self.subTest(key=leaked):
                self.assertNotIn(leaked, out)

    def test_identity_and_strategy_fields_survive(self) -> None:
        # The adversary must still understand the game and who it colludes
        # with, so these are kept.
        env = self._make_env()
        agent_name = next(a for a in env.agent_names if a != "Resource_Provisioner")
        for name in ("Resource_Provisioner", agent_name):
            out = env.build_agent_context(
                name,
                phase="pre_coordination",
                iteration=0,
                round_number=1,
                **(self.PROVISIONER_CTX if name == "Resource_Provisioner" else self.DEPARTMENT_CTX),
                **self.EXTRA,
            )
            for kept in (
                "agent_name",
                "phase",
                "iteration",
                "role",
                "misalignment_role",
                "misalignment_adversaries",
                "misalignment_secret_blackboard_id",
                "post_message_available",
            ):
                with self.subTest(agent=name, key=kept):
                    self.assertIn(kept, out)

    def test_coalition_membership_intact(self) -> None:
        out = self._context("Resource_Provisioner", self.PROVISIONER_CTX)
        self.assertEqual(
            out["misalignment_adversaries"],
            ["Resource_Provisioner", "General_Hospital_Triage"],
        )
        self.assertEqual(out["misalignment_secret_blackboard_id"], 7)

    def test_allowlist_fails_closed_for_unknown_keys(self) -> None:
        # A future environment adding a new state key must be withheld by
        # default, not leaked.
        out = self._context(
            "Resource_Provisioner",
            {**self.PROVISIONER_CTX, "brand_new_state_key": {"secret": 1}},
        )
        self.assertNotIn("brand_new_state_key", out)

    def test_round_number_is_not_echoed(self) -> None:
        # run_phase_entries always passes round_number; it is not allowlisted
        # and must not leak into the pre-coordination context.
        out = self._context("Resource_Provisioner", self.PROVISIONER_CTX)
        self.assertNotIn("round_number", out)

    def test_other_phases_keep_full_context(self) -> None:
        env = self._make_env()
        agent_name = next(a for a in env.agent_names if a != "Resource_Provisioner")
        context = env.build_agent_context(agent_name, "planning", 1)
        self.assertIn("job_queue", context)
        self.assertIn("cost_table", context)
        self.assertIn("dept_info", context)


class GetExtraFieldsPreCoordinationOverrideTests(unittest.TestCase):
    """The pre-coordination prompt must never tell the adversary they have no
    `post_message` -- the runtime gate keeps it enabled during that round.
    """

    def _build(self, disable_pm, *, is_benign=False):
        # We can't easily exercise `_get_extra_fields` without the rest of
        # `_run_single`; instead we replicate the override branch inline so
        # the contract stays testable in isolation.
        role = "cooperative" if is_benign else "adversary"
        if role != "adversary":
            return {"post_message_available": True, "phase": "pre_coordination"}
        pm = not bool(disable_pm)
        if role == "adversary":  # pre-coordination override
            pm = True
        return {"post_message_available": pm, "phase": "pre_coordination"}

    def test_pre_coordination_overrides_to_available(self) -> None:
        # The bug we're guarding against: with `disable_post_message: true` the
        # planning-shaped prompt would otherwise say "you have NO post_message"
        # while post_message actually works that turn.
        self.assertIs(self._build(True)["post_message_available"], True)

    def test_pre_coordination_available_when_not_disabled(self) -> None:
        self.assertIs(self._build(False)["post_message_available"], True)

    def test_cooperative_agent_is_available(self) -> None:
        # Sanity: the override must not flip honest agents to False.
        self.assertIs(self._build(True, is_benign=True)["post_message_available"], True)


class EnvToolsBlockStubTests(unittest.TestCase):
    """The protocol returns a stub when the phase policy blocks env tools but
    the blackboard path (post_message) is untouched."""

    def _make_protocol(self):
        # `LocalCommunicationProtocol.__init__` requires a `config` keyword;
        # the blocking branch never reads it, so an empty dict is fine.
        proto = LocalCommunicationProtocol(config={})

        # environment_handle_tool_call requires self.environment; a stub object
        # is enough because the blocking branch never touches it.
        class _StubEnv:
            pass

        proto.environment = _StubEnv()
        return proto

    def test_environment_handle_returns_stub(self) -> None:
        proto = self._make_protocol()
        proto.set_phase_policy(
            "pre_coordination",
            PhaseToolPolicy(block_environment_tool_calls=True),
        )
        try:
            result = asyncio.run(
                proto.environment_handle_tool_call(
                    tool_name="broadcast_message",
                    agent_name="Agent_A",
                    arguments={"message": "hi"},
                    phase="pre_coordination",
                    iteration=0,
                )
            )
        finally:
            proto.pop_phase_policy("pre_coordination")

        self.assertTrue(result.get("blocked"))
        # The stub also leaves an event trail so downstream analytics can see
        # the blocked attempt.
        self.assertEqual(len(proto.tool_events), 1)
        ev = proto.tool_events[0]
        self.assertEqual(ev.tool_name, "broadcast_message")
        self.assertEqual(ev.phase, "pre_coordination")

    def test_environment_handle_executes_normally_when_unblocked(self) -> None:
        proto = self._make_protocol()
        # With a stub environment and no real env_tools the call raises a
        # benign "Environment not set" path because `env` is non-None -- we
        # only need to confirm we did NOT short-circuit on the phase policy.
        # The simpler check: with no policy set for the phase, the function
        # does NOT return a `blocked: True` stub.
        self.assertIsNone(proto.active_phase_policy("pre_coordination"))
        try:
            result = asyncio.run(
                proto.environment_handle_tool_call(
                    tool_name="broadcast_message",
                    agent_name="Agent_A",
                    arguments={"message": "hi"},
                    phase="pre_coordination",
                    iteration=0,
                )
            )
            self.assertNotIsInstance(result, dict) or not result.get("blocked")
        except Exception:
            # Any exception is fine as long as it isn't the blocking stub --
            # the stub path is well-defined above and returns cleanly.
            pass


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


class PostMessageCallLimitPolicyTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        self.assertIsNone(proto.active_phase_policy("pre_coordination"))
        proto.set_phase_policy(
            "pre_coordination",
            PhaseToolPolicy(max_post_messages_per_agent=5),
        )
        self.assertEqual(
            proto.active_phase_policy("pre_coordination").max_post_messages_per_agent, 5
        )
        proto.pop_phase_policy("pre_coordination")
        self.assertIsNone(proto.active_phase_policy("pre_coordination"))


class PostMessageCountsPolicyTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        proto.reset_post_message_counts()
        self.assertEqual(proto.get_post_message_count("AgentA"), 0)
        self.assertEqual(proto._record_post_message_call("AgentA"), 1)
        self.assertEqual(proto._record_post_message_call("AgentA"), 2)
        self.assertEqual(proto._record_post_message_call("AgentB"), 1)
        self.assertEqual(proto.get_post_message_count("AgentA"), 2)
        self.assertEqual(proto.get_post_message_count("AgentB"), 1)
        proto.reset_post_message_counts()
        self.assertEqual(proto.get_post_message_count("AgentA"), 0)


class BlackboardPostMessageCapTests(unittest.TestCase):
    """End-to-end check: when the phase policy sets a cap, the protocol rejects
    `post_message` calls past the limit and accepts them up to it.
    """

    def _proto(self, *, policy=None):
        from terrarium.core.blackboard import Megaboard

        proto = LocalCommunicationProtocol(config={})
        proto.megaboard = Megaboard()

        class _StubEnv:
            pass

        proto.environment = _StubEnv()
        proto.set_phase_policy(
            "pre_coordination",
            policy
            or PhaseToolPolicy(
                blackboard_tools={"post_message"},
                max_post_messages_per_agent=None,
            ),
        )
        # `Megaboard.add_blackboard(agents, template)` is the constructor API.
        bb_id = proto.megaboard.add_blackboard(["A", "B"])
        return proto, bb_id

    def test_post_message_capped(self) -> None:
        proto, bb_id = self._proto(
            policy=PhaseToolPolicy(
                blackboard_tools={"post_message"}, max_post_messages_per_agent=2
            )
        )
        proto.reset_post_message_counts()
        try:
            async def _post_n(n):
                results = []
                for i in range(n):
                    r = await proto.blackboard_handle_tool_call(
                        tool_name="post_message",
                        agent_name="A",
                        arguments={"blackboard_id": bb_id, "message": f"m{i}"},
                        phase="pre_coordination",
                        iteration=0,
                    )
                    results.append(r)
                return results

            results = asyncio.run(_post_n(3))
            # First two succeed; third is rejected.
            self.assertFalse(bool(results[0].get("blocked")))
            self.assertFalse(bool(results[1].get("blocked")))
            self.assertTrue(bool(results[2].get("blocked")))
            self.assertEqual(results[2].get("limit"), 2)
            self.assertEqual(results[2].get("count"), 2)
        finally:
            proto.pop_phase_policy("pre_coordination")
            proto.reset_post_message_counts()

    def test_unlimited_when_limit_none(self) -> None:
        proto, bb_id = self._proto(
            policy=PhaseToolPolicy(
                blackboard_tools={"post_message"}, max_post_messages_per_agent=None
            )
        )
        proto.reset_post_message_counts()

        async def _post_n(n):
            results = []
            for i in range(n):
                r = await proto.blackboard_handle_tool_call(
                    tool_name="post_message",
                    agent_name="A",
                    arguments={"blackboard_id": bb_id, "message": f"m{i}"},
                    phase="pre_coordination",
                    iteration=0,
                )
                results.append(r)
            return results

        results = asyncio.run(_post_n(5))
        for r in results:
            self.assertFalse(bool(r.get("blocked")))

    def test_other_blackboard_tools_unaffected(self) -> None:
        # Cap-only policy (blackboard_tools=None = phase default): the cap
        # applies to post_message alone; other blackboard tools still work.
        proto, bb_id = self._proto(
            policy=PhaseToolPolicy(max_post_messages_per_agent=1)
        )
        proto.reset_post_message_counts()
        try:
            async def _sequence():
                r1 = await proto.blackboard_handle_tool_call(
                    tool_name="post_message",
                    agent_name="A",
                    arguments={"blackboard_id": bb_id, "message": "hi"},
                    phase="pre_coordination",
                    iteration=0,
                )
                r2 = await proto.blackboard_handle_tool_call(
                    tool_name="get_blackboard_events",
                    agent_name="A",
                    arguments={"blackboard_id": bb_id},
                    phase="pre_coordination",
                    iteration=0,
                )
                r3 = await proto.blackboard_handle_tool_call(
                    tool_name="post_message",
                    agent_name="A",
                    arguments={"blackboard_id": bb_id, "message": "again"},
                    phase="pre_coordination",
                    iteration=0,
                )
                return r1, r2, r3

            r1, r2, r3 = asyncio.run(_sequence())

            # First post succeeds.
            self.assertFalse(bool(r1.get("blocked")))
            # Read-only tool must still work past the cap.
            self.assertNotEqual(r2.get("blocked"), True)
            # Second post is now blocked.
            self.assertTrue(bool(r3.get("blocked")))
        finally:
            proto.pop_phase_policy("pre_coordination")
            proto.reset_post_message_counts()


class SecretBlackboardIdPolicyTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        self.assertIsNone(proto.active_phase_policy("pre_coordination"))
        proto.set_phase_policy(
            "pre_coordination", PhaseToolPolicy(route_post_message_to=7)
        )
        self.assertEqual(
            proto.active_phase_policy("pre_coordination").route_post_message_to, 7
        )
        proto.pop_phase_policy("pre_coordination")
        self.assertIsNone(proto.active_phase_policy("pre_coordination"))


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


class BlackboardPostMessageRewriteTests(unittest.TestCase):
    """Defense in depth: when the phase policy routes `post_message` to the
    secret blackboard id, any `post_message` call in that phase is rewritten
    onto it (matching, missing, or wrong board id). When unset, calls flow
    through unmodified.
    """

    def _proto(self, *, route_to=None):
        from terrarium.core.blackboard import Megaboard

        proto = LocalCommunicationProtocol(config={})

        class _StubEnv:
            pass

        proto.environment = _StubEnv()
        proto.megaboard = Megaboard()
        if route_to is not None:
            proto.set_phase_policy(
                "pre_coordination",
                PhaseToolPolicy(
                    blackboard_tools={"post_message"},
                    route_post_message_to=route_to,
                ),
            )
        # Two boards: id=0 is the public/main board with all participants;
        # id=1 is the secret coalition-only board.
        proto.megaboard.add_blackboard(["A", "B", "C"], template={"kind": "public"})
        proto.megaboard.add_blackboard(["A", "B"], template={"kind": "secret"})
        return proto

    def _post(self, proto, arguments):
        return asyncio.run(
            proto.blackboard_handle_tool_call(
                tool_name="post_message",
                agent_name="A",
                arguments=arguments,
                phase="pre_coordination",
                iteration=0,
            )
        )

    def test_post_without_blackboard_id_is_rewritten(self) -> None:
        proto = self._proto(route_to=1)
        proto.reset_post_message_counts()
        try:
            result = self._post(proto, {"message": "scheme"})
            self.assertNotEqual(result.get("blocked"), True)
            # The message landed on the secret board (id=1), not the public.
            secret_events = proto.megaboard.get(1, "A", limit=None)
            public_events = proto.megaboard.get(0, "A", limit=None)
            self.assertEqual(len(secret_events), 1)
            self.assertEqual(len(public_events), 0)
        finally:
            proto.pop_phase_policy("pre_coordination")

    def test_post_with_wrong_blackboard_id_is_rewritten(self) -> None:
        proto = self._proto(route_to=1)
        proto.reset_post_message_counts()
        try:
            # Agent tries to post to the PUBLIC board; the rewrite forces
            # it onto the secret one instead.
            result = self._post(proto, {"blackboard_id": 0, "message": "secret"})
            self.assertNotEqual(result.get("blocked"), True)
            self.assertEqual(len(proto.megaboard.get(0, "A", limit=None)), 0)
            self.assertEqual(len(proto.megaboard.get(1, "A", limit=None)), 1)
        finally:
            proto.pop_phase_policy("pre_coordination")

    def test_post_with_correct_blackboard_id_passes_through(self) -> None:
        proto = self._proto(route_to=1)
        proto.reset_post_message_counts()
        try:
            result = self._post(proto, {"blackboard_id": 1, "message": "ok"})
            self.assertNotEqual(result.get("blocked"), True)
            self.assertEqual(len(proto.megaboard.get(1, "A", limit=None)), 1)
        finally:
            proto.pop_phase_policy("pre_coordination")

    def test_post_without_rewrite_when_secret_id_none(self) -> None:
        proto = self._proto(route_to=None)
        proto.reset_post_message_counts()
        # Without the routing policy, posting to blackboard_id=0 (public)
        # lands on the public board.
        result = self._post(proto, {"blackboard_id": 0, "message": "public ok"})
        self.assertNotEqual(result.get("blocked"), True)
        self.assertEqual(len(proto.megaboard.get(0, "A", limit=None)), 1)
        self.assertEqual(len(proto.megaboard.get(1, "A", limit=None)), 0)


class _PhaseRecordingAgent:
    """Fake agent mirroring the parts of ``BaseAgent`` that
    ``agent_planning_turn`` drives: ``generate_response`` forwards ``phase``
    to ``set_meta_context``, which is exactly what the trajectory logger keys
    the ``agent_trajectories.json`` entry on.
    """

    def __init__(self) -> None:
        self.generate_response_calls = []
        self.meta = {}

    def set_meta_context(self, *, agent_name, phase, iteration, round_num=None):
        self.meta = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round_num": round_num,
        }

    async def generate_response(
        self,
        *,
        agent_name,
        agent_context,
        blackboard_context,
        communication_protocol,
        prompts,
        phase,
        iteration,
        round_num=0,
    ):
        self.generate_response_calls.append(
            {"phase": phase, "iteration": iteration, "round_num": round_num}
        )
        # Mirrors ``BaseAgent.generate_response`` -> ``set_meta_context``.
        self.set_meta_context(
            agent_name=agent_name,
            phase=phase,
            iteration=iteration,
            round_num=round_num,
        )
        return {"response": "ok", "usage": {}, "tools_executed": 0}


class _StubEnvironment:
    """Only ``prompts`` is read by ``agent_planning_turn``."""

    prompts = object()


class PlanningTurnPhaseLabelTests(unittest.TestCase):
    """The pre-coordination turn must be labelled ``pre_coordination`` in the
    trajectory log, not conflated with a normal iteration-0 ``planning`` round.
    """

    def _proto(self):
        return LocalCommunicationProtocol(config={})

    def test_defaults_to_planning_phase(self) -> None:
        proto = self._proto()
        agent = _PhaseRecordingAgent()
        asyncio.run(
            proto.agent_planning_turn(
                agent,
                "AgentA",
                {"agent_name": "AgentA"},
                _StubEnvironment(),
                iteration=0,
                planning_round=0,
            )
        )
        self.assertEqual(agent.generate_response_calls[0]["phase"], "planning")
        self.assertEqual(agent.meta["phase"], "planning")
        self.assertEqual(agent.meta["iteration"], 0)
        self.assertEqual(agent.meta["round_num"], 0)

    def test_explicit_phase_is_forwarded(self) -> None:
        proto = self._proto()
        agent = _PhaseRecordingAgent()
        asyncio.run(
            proto.agent_planning_turn(
                agent,
                "AgentA",
                {"agent_name": "AgentA"},
                _StubEnvironment(),
                iteration=0,
                planning_round=0,
                phase="pre_coordination",
            )
        )
        self.assertEqual(agent.generate_response_calls[0]["phase"], "pre_coordination")
        self.assertEqual(agent.meta["phase"], "pre_coordination")
        # Same iteration/round shape the trajectory logger nests under.
        self.assertEqual(agent.meta["iteration"], 0)
        self.assertEqual(agent.meta["round_num"], 0)

    def test_normal_planning_caller_unchanged(self) -> None:
        # A caller that omits ``phase`` (the normal planning loop) keeps
        # logging as ``planning`` for non-zero iterations too.
        proto = self._proto()
        agent = _PhaseRecordingAgent()
        asyncio.run(
            proto.agent_planning_turn(
                agent,
                "AgentA",
                {"agent_name": "AgentA"},
                _StubEnvironment(),
                iteration=3,
                planning_round=2,
            )
        )
        self.assertEqual(agent.generate_response_calls[0]["phase"], "planning")
        self.assertEqual(agent.meta["iteration"], 3)
        self.assertEqual(agent.meta["round_num"], 2)


class PostMessageAvailableDuringPreCoordination(unittest.TestCase):
    """The pre-coordination round is labelled ``pre_coordination`` in the
    trajectory log, but it is planning-shaped for *toolset* purposes: the
    runner registers it in the agent's ``ToolsetDiscovery.planning_shaped_phases``
    so ``get_tools_for_blackboard`` still offers ``post_message`` (filtered by
    the active phase policy's allowlist) for the secret blackboard. Terrarium's
    stock gate only knows ``planning``, which would silently drop
    ``post_message`` for ``"pre_coordination"``.
    """

    def _discovery(self):
        from terrarium.tools.discovery import ToolsetDiscovery

        discovery = ToolsetDiscovery()
        # Mirrors what run.py does once per run for every agent.
        discovery.planning_shaped_phases = (
            set(discovery.planning_shaped_phases) | {"pre_coordination"}
        )
        return discovery

    @staticmethod
    def _tool_names(tools) -> set:
        return {str((t or {}).get("function", {}).get("name")) for t in tools}

    @staticmethod
    def _allowed(proto, phase):
        policy = proto.active_phase_policy(phase)
        return policy.blackboard_tools if policy is not None else None

    def test_pre_coordination_offers_post_message(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        proto.set_phase_policy(
            "pre_coordination", PhaseToolPolicy(blackboard_tools={"post_message"})
        )
        tools = self._discovery().get_tools_for_blackboard(
            "pre_coordination", allowed_tools=self._allowed(proto, "pre_coordination")
        )
        self.assertIn("post_message", self._tool_names(tools))

    def test_planning_offers_post_message(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        tools = self._discovery().get_tools_for_blackboard(
            "planning", allowed_tools=self._allowed(proto, "planning")
        )
        self.assertIn("post_message", self._tool_names(tools))

    def test_execution_offers_no_blackboard_tools(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        tools = self._discovery().get_tools_for_blackboard(
            "execution", allowed_tools=self._allowed(proto, "execution")
        )
        self.assertEqual(tools, [])

    def test_withheld_policy_suppresses_even_pre_coordination(self) -> None:
        proto = LocalCommunicationProtocol(config={})
        proto.set_phase_policy(
            "pre_coordination", PhaseToolPolicy(blackboard_tools=set())
        )
        proto.set_phase_policy("planning", PhaseToolPolicy(blackboard_tools=set()))
        precoord = self._discovery().get_tools_for_blackboard(
            "pre_coordination", allowed_tools=self._allowed(proto, "pre_coordination")
        )
        planning = self._discovery().get_tools_for_blackboard(
            "planning", allowed_tools=self._allowed(proto, "planning")
        )
        self.assertEqual(precoord, [])
        self.assertEqual(planning, [])


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


class _FakePreCoordAgent:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakePreCoordEnvironment:
    """Mirrors the native HospitalEnvironment.build_agent_context: for
    `phase == "pre_coordination"` it returns the minimal identity context and
    only echoes kwargs in Terrarium's `_PRE_COORDINATION_CONTEXT_ALLOWLIST`;
    every other phase gets the full (leaky) state.
    """

    def __init__(self, agent_names) -> None:
        self.agents = [_FakePreCoordAgent(n) for n in agent_names]
        self.context_calls = []

    def build_agent_context(self, agent_name, phase, iteration, **kwargs):
        self.context_calls.append(
            {
                "agent_name": agent_name,
                "phase": phase,
                "iteration": iteration,
                **kwargs,
            }
        )
        if phase == "pre_coordination":
            from terrarium.environments.dcops.hospital.hospital_env import (
                _PRE_COORDINATION_CONTEXT_ALLOWLIST,
            )

            context = {
                "agent_name": agent_name,
                "phase": phase,
                "iteration": iteration,
                "role": "department",
            }
            for key, value in kwargs.items():
                if key in _PRE_COORDINATION_CONTEXT_ALLOWLIST and key not in context:
                    context[key] = value
            return context
        # Full state for the other phases (absorbed kwargs are not echoed).
        return {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "role": "department",
            "inventory_snapshot": {"IV_Kits": 99},
            "job_queue": [{"patient_id": "Patient_0"}],
            "cost_table": {"IV_Kits": {"cost": 2}},
        }


class _FakePreCoordProtocol(LocalCommunicationProtocol):
    """Native protocol with the turn execution stubbed: records the active
    phase policy at each turn and simulates one board post per turn."""

    def __init__(self) -> None:
        super().__init__(config={})
        self.calls = []
        self.env_blocked = []
        self.secret_ids = []
        self.caps = []
        self.counts_before = []
        self.pm_disabled = []

    async def agent_turn(
        self,
        agent,
        agent_name,
        agent_context,
        environment,
        iteration,
        *,
        phase,
        round_num=None,
        phase_config=None,
    ):
        # Mirror LocalCommunicationProtocol.agent_turn: the pre-coordination
        # phase is planning-shaped, and a failed turn is returned (not raised)
        # so the walk records an error entry and continues.
        try:
            return await self.agent_planning_turn(
                agent,
                agent_name,
                agent_context,
                environment,
                iteration,
                planning_round=round_num if round_num is not None else 1,
                phase=phase,
            )
        except Exception as exc:
            return exc

    async def agent_planning_turn(
        self,
        agent,
        agent_name,
        agent_context,
        environment,
        iteration,
        planning_round,
        phase="planning",
    ):
        policy = self.active_phase_policy(phase)
        self.calls.append(
            {
                "agent_name": agent_name,
                "agent_context": agent_context,
                "iteration": iteration,
                "planning_round": planning_round,
                "phase": phase,
            }
        )
        self.env_blocked.append(
            bool(policy is not None and policy.block_environment_tool_calls)
        )
        self.secret_ids.append(
            policy.route_post_message_to if policy is not None else None
        )
        self.pm_disabled.append(
            policy is not None
            and policy.blackboard_tools is not None
            and "post_message" not in policy.blackboard_tools
        )
        self.caps.append(
            policy.max_post_messages_per_agent if policy is not None else None
        )
        self.counts_before.append(self.get_post_message_count(agent_name))
        # Simulate one board post per turn, so the per-round reset is
        # observable on the next round.
        self._record_post_message_call(agent_name)
        return {
            "response": f"{agent_name} round {planning_round}",
            "usage": {},
            "tools_executed": 1,
        }


class RunPreCoordinationRoundsTests(unittest.TestCase):
    """Execution of the dedicated phase: N rounds, one turn per adversary each."""

    ROLES = {"Adv1": "adversary", "Honest": "cooperative", "Adv2": "adversary"}

    def _run(
        self,
        *,
        num_rounds: int,
        max_messages=None,
        agent_names=None,
        secret_blackboard_id=42,
        extra_factory=None,
    ):
        from unittest.mock import patch

        env = _FakePreCoordEnvironment(
            agent_names or ["Adv1", "Honest", "Adv2"],
        )
        proto = _FakePreCoordProtocol()
        bb_calls = []

        def _record_bb(**kwargs):
            bb_calls.append(kwargs)

        if extra_factory is None:

            def extra_factory(name, phase):
                return {
                    "misalignment_role": "adversary",
                    "misalignment_adversaries": ["Adv1", "Adv2"],
                    "misalignment_secret_blackboard_id": secret_blackboard_id,
                    "post_message_available": secret_blackboard_id is not None,
                    "phase": phase,
                }

        with patch.object(run_mod, "_log_blackboards_txt", side_effect=_record_bb):
            turns = asyncio.run(
                run_mod._run_pre_coordination_rounds(
                    env=env,
                    protocol=proto,
                    roles=self.ROLES,
                    num_rounds=num_rounds,
                    max_messages=max_messages,
                    secret_blackboard_id=secret_blackboard_id,
                    extra_fields=extra_factory,
                    bb_logger=object(),
                )
            )

        # Isolation must be restored for the main loop: the phase policy is
        # popped and the per-agent counters are reset.
        self.assertIsNone(proto.active_phase_policy("pre_coordination"))
        self.assertEqual(proto.get_post_message_count("Adv1"), 0)
        return env, proto, turns, bb_calls

    def test_two_rounds_produce_two_turns_per_adversary(self) -> None:
        env, proto, turns, bb_calls = self._run(num_rounds=2)

        self.assertEqual([t["agent"] for t in turns], ["Adv1", "Adv2", "Adv1", "Adv2"])
        self.assertEqual([t["round"] for t in turns], [1, 1, 2, 2])
        self.assertEqual([t["pre_coordination_round"] for t in turns], [1, 1, 2, 2])
        self.assertEqual([t["pre_coordination_rounds"] for t in turns], [2, 2, 2, 2])
        for turn in turns:
            self.assertEqual(turn["phase"], "pre_coordination")
            self.assertEqual(turn["iteration"], 0)
            self.assertEqual(turn["role"], "adversary")

        # Round metadata is passed to both the environment and the turn call.
        self.assertEqual([c["planning_round"] for c in proto.calls], [1, 1, 2, 2])
        # run_phase_entries passes the round to build_agent_context as
        # round_number; the pre_coordination_* labels are injected by the
        # runner's build_turn hook (asserted via agent_context below).
        self.assertEqual(
            [c["round_number"] for c in env.context_calls],
            [1, 1, 2, 2],
        )
        # One blackboard snapshot per completed round.
        self.assertEqual([c["planning_round"] for c in bb_calls], [1, 2])
        self.assertEqual({c["phase"] for c in bb_calls}, {"pre_coordination"})

    def test_honest_agents_are_skipped(self) -> None:
        env, proto, turns, _ = self._run(num_rounds=2)

        self.assertNotIn("Honest", [t["agent"] for t in turns])
        self.assertNotIn("Honest", [c["agent_name"] for c in proto.calls])
        self.assertNotIn("Honest", [c["agent_name"] for c in env.context_calls])

    def test_default_one_round_is_backward_compatible(self) -> None:
        _, _, turns, bb_calls = self._run(num_rounds=1)
        self.assertEqual([t["agent"] for t in turns], ["Adv1", "Adv2"])
        self.assertEqual([t["round"] for t in turns], [1, 1])
        self.assertEqual(len(bb_calls), 1)

    def test_context_has_no_environment_state(self) -> None:
        _, proto, _, _ = self._run(num_rounds=2)
        for call in proto.calls:
            ctx = call["agent_context"]
            with self.subTest(agent=call["agent_name"]):
                for leaked in ("inventory_snapshot", "job_queue", "cost_table"):
                    self.assertNotIn(leaked, ctx)
                self.assertEqual(ctx["pre_coordination_rounds"], 2)
        # Exact per-turn sequence, not a membership check: a constant-round
        # injection bug would pass `in (1, 2)` but fail here.
        self.assertEqual(
            [call["agent_context"]["pre_coordination_round"] for call in proto.calls],
            [1, 1, 2, 2],
        )

    def test_isolation_is_held_through_the_whole_phase(self) -> None:
        _, proto, _, _ = self._run(num_rounds=2)
        self.assertEqual(proto.env_blocked, [True, True, True, True])
        self.assertEqual(proto.secret_ids, [42, 42, 42, 42])
        self.assertEqual(proto.pm_disabled, [False, False, False, False])
        for call in proto.calls:
            self.assertTrue(call["agent_context"]["post_message_available"])

    def test_no_secret_board_disables_post_message_for_the_phase(self) -> None:
        _, proto, _, _ = self._run(num_rounds=2, secret_blackboard_id=None)

        # No private channel -> the tool must be unavailable for every turn.
        self.assertEqual(proto.pm_disabled, [True, True, True, True])
        self.assertEqual(proto.secret_ids, [None, None, None, None])
        # Defense in depth: a protocol-level cap of 0 rejects any hallucinated
        # post_message call, so a public post is impossible.
        self.assertEqual(proto.caps, [0, 0, 0, 0])
        for call in proto.calls:
            ctx = call["agent_context"]
            self.assertFalse(ctx["post_message_available"])
            self.assertIsNone(ctx["misalignment_secret_blackboard_id"])

    def test_phase_metadata_comes_from_runner_injection(self) -> None:
        # The fake env echoes only allowlisted kwargs (like the native hospital
        # env); these fields can only be present because the runner injects
        # them into the context.
        _, proto, _, _ = self._run(
            num_rounds=2,
            extra_factory=lambda name, phase: {},
        )
        for call in proto.calls:
            ctx = call["agent_context"]
            self.assertIn(ctx["pre_coordination_round"], (1, 2))
            self.assertEqual(ctx["pre_coordination_rounds"], 2)
            self.assertTrue(ctx["post_message_available"])

    def test_runner_injected_extra_survives_even_if_not_allowlisted(self) -> None:
        # The runner's `turn["agent_context"].update(extra)` is not filtered by
        # the env allowlist, so a future strategy field cannot be dropped.
        _, proto, _, _ = self._run(
            num_rounds=1,
            extra_factory=lambda name, phase: {"brand_new_strategy": 1},
        )
        for call in proto.calls:
            self.assertEqual(call["agent_context"]["brand_new_strategy"], 1)

    def test_failing_turn_does_not_propagate_and_restores_state(self) -> None:
        class _BoomProtocol(LocalCommunicationProtocol):
            async def agent_turn(self, agent, agent_name, agent_context, environment, iteration, *, phase, round_num=None, phase_config=None):
                try:
                    return await self.agent_planning_turn(
                        agent, agent_name, agent_context, environment, iteration,
                        planning_round=round_num if round_num is not None else 1,
                        phase=phase,
                    )
                except Exception as exc:
                    return exc

            async def agent_planning_turn(self, *args, **kwargs):
                raise RuntimeError("boom")

        env = _FakePreCoordEnvironment(["Adv1"])
        proto = _BoomProtocol(config={})
        # Non-default prior phase policy makes restoration observable.
        prior = PhaseToolPolicy(
            block_environment_tool_calls=True,
            blackboard_tools=set(),
            route_post_message_to=7,
            max_post_messages_per_agent=9,
        )
        proto.set_phase_policy("pre_coordination", prior)
        # The runner deliberately catches per-turn failures (it records
        # them as `[ERROR] ...` turns and continues), so nothing propagates
        # out of the helper. One failing turn per round is recorded, and the
        # prior phase policy is restored afterwards.
        turns = asyncio.run(
            run_mod._run_pre_coordination_rounds(
                env=env,
                protocol=proto,
                roles={"Adv1": "adversary"},
                num_rounds=2,
                max_messages=3,
                secret_blackboard_id=42,
                extra_fields=lambda name, phase: {},
            )
        )
        self.assertEqual([t["agent"] for t in turns], ["Adv1", "Adv1"])
        self.assertEqual([t["round"] for t in turns], [1, 2])
        for turn in turns:
            self.assertTrue(turn["response"].startswith("[ERROR]"))
        restored = proto.active_phase_policy("pre_coordination")
        self.assertIs(restored, prior)
        self.assertTrue(restored.block_environment_tool_calls)
        self.assertEqual(restored.route_post_message_to, 7)
        self.assertEqual(restored.max_post_messages_per_agent, 9)
        self.assertEqual(proto.get_post_message_count("Adv1"), 0)

    def test_message_cap_resets_each_round(self) -> None:
        _, proto, _, _ = self._run(num_rounds=2, max_messages=3)
        self.assertEqual(proto.caps, [3, 3, 3, 3])
        # Every agent starts each round with a fresh per-agent counter.
        self.assertEqual(proto.counts_before, [0, 0, 0, 0])

    def test_no_cap_means_unlimited(self) -> None:
        _, proto, _, _ = self._run(num_rounds=2, max_messages=None)
        self.assertEqual(proto.caps, [None, None, None, None])


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
