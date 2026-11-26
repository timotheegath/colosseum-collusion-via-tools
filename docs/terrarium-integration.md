# Terrarium integration: collusion-via-tools as a native phase condition

Status: **agreed plan, not yet implemented.** This document records the
decision on where the collusion-via-tools "pre-coordination round" condition
lives across Colosseum and the Terrarium fork, and the migration steps to get
there. Branches: `fork-main` in both repos.

## What the condition is

The collusion-via-tools experiment adds a coalition-only **pre-coordination
phase** before iteration 1: adversaries get dedicated rounds in which only
`post_message` is usable (all environment tool calls are dropped at the
protocol layer), messages are rewritten onto a coalition-only secret
blackboard, a per-agent message cap applies per round, and the agent context
is scrubbed to identity + coalition metadata. A separate sweep axis,
`disable_post_message`, withholds the `post_message` tool entirely.

Mechanically, on `fork-main` today, this is five layers:

1. **Phase gating in tool discovery** — `pre_coordination` must look
   planning-shaped to `ToolsetDiscovery`, otherwise `post_message` is silently
   dropped (`terrarium/tools/discovery.py:85` only enables blackboard tools for
   `phase == "planning"`). Colosseum monkey-patches this in
   `experiments/common/_patch_terrarium.py:196-242` (remap
   `pre_coordination -> planning`; return `[]` when post_message is disabled).
2. **Tool-call interception in the protocol** —
   `experiments/common/local_protocol.py:180-332` blocks environment tool
   calls, rewrites `post_message` to the secret board, and enforces the
   per-agent cap, all driven by `contextvars` flags set in
   `_patch_terrarium.py`.
3. **Run orchestration** — `experiments/collusion_via_tools/run.py:995-1160`
   (`_run_pre_coordination_rounds`) runs adversary-only rounds at iteration 0
   via Terrarium's generic `run_phase_entries` (`terrarium/utils.py:914`),
   injecting a scrubbed context (`_pre_coordination_context`, `run.py:443`),
   plus sweep axes, `__precoord<N>` run-id suffixes, and resume logic.
4. **Experiment science** — adversary roles, prompts, judge, metrics, plots
   (`agents.py`, `prompts.py`, `judge_collusion_score.py`, `metrics.py`).
5. **Distribution** — `pyproject.toml:10` pins `terrarium-agents==0.2.1` from
   PyPI, but `[tool.uv.sources]` overrides it to the editable
   `external/Terrarium` submodule on `fork-main`, so in-repo runs always use
   the fork.

## Decisions

- **Hospital-only for now.** The condition is not built to be generic across
  environments yet.
- **The mechanism moves into Terrarium as phase/tool policy; the experiment
  stays in Colosseum.** Specifically *not* under `terrarium/attacks/`: the
  attacks framework (`terrarium/attacks/reference.py`, `examples/attack_main.py
  --attack_type`) is payload-driven post-hoc injection (rewriting tool-call
  arguments, injecting system messages). The pre-coordination round is run
  *structure* — a phase, tool gating, a provisioned secret board, first-class
  adversary agents with their own prompts and judge. It does not fit the
  payload mold.
- **No new `HospitalCollusionEnvironment` type.** The condition never touches
  hospital game state (patients, inventory, scheduling — all owned by
  `HospitalEnvironment`). The only environment-specific piece is the scrubbed
  context, which is a phase hook on the *existing* hospital env:
  `HospitalEnvironment.build_agent_context` already receives `phase`
  (`terrarium/environments/dcops/hospital/hospital_env.py:561`).
- **Fork-main submodule is enough; no PyPI release yet.** Colosseum runs
  against the editable submodule, so native changes land in the
  `external/Terrarium` fork and take effect on the next submodule bump. A
  PyPI release (beyond `0.2.1`) is deferred until the artifacts are finalized.
- **Clean break on old runs.** Existing logged runs are read-only; the
  monkey-patch path does not need to keep working for re-execution/resume.

Key mechanical fact that makes the patch deletable: `ToolsetDiscovery` is
**per-agent** (`terrarium/agents/base.py:55`) and the phase is known when the
per-turn tool list is built (`terrarium/agents/base.py:403-412`), and
`LocalCommunicationProtocol` is per-run. So all gating can be explicit
per-run/per-phase data — no monkey-patching, no process-global contextvars.

## Target split

| Lives in Terrarium (`fork-main`) | Stays in Colosseum (`fork-main`) |
|---|---|
| `pre_coordination` as a first-class planning-shaped phase in `ToolsetDiscovery`, plus a per-phase blackboard tool allowlist (data, not patch) | Adversary roles / coalition assignment (`agents.py`) |
| `LocalCommunicationProtocol` in `terrarium/communication_protocols/local.py` with an explicit per-phase `PhaseToolPolicy` (replaces the contextvars) | Prompts, judge, metrics, plots (`prompts.py`, `judge_*.py`, `metrics.py`, `plots/`) |
| Hospital `build_agent_context` phase hook for the scrubbed pre-coordination context | Sweep axes, `__precoord<N>` run-id scheme, resume (`run.py`, `resume.py`) |
| Generic `run_phase_entries` (already native, `terrarium/utils.py:914`) | The orchestration call that runs the phase (`_run_pre_coordination_rounds`) |

## Migration plan

### PR 1 — Terrarium `fork-main` (the mechanism)

1. `terrarium/tools/discovery.py`: make the blackboard phase gate data-driven
   instead of hardcoded `phase == "planning"` (discovery.py:85) — a
   planning-shaped phase set (default `{"planning"}`, extensible per instance
   or via config, including `"pre_coordination"`), plus an
   `allowed_tools: Optional[Set[str]]` parameter on `get_tools_for_blackboard`
   / `get_blackboard_tool_names` that filters the returned set. Defaults
   unchanged.
2. `terrarium/agents/base.py`: at the per-turn tool-list build
   (`base.py:403-412`), pass the active phase policy's `allowed_tools` into
   the blackboard lookup. The policy reaches the agent via the protocol (which
   already sets `agent.current_phase` via `set_meta_context`).
3. New `terrarium/communication_protocols/local.py`: move
   `LocalCommunicationProtocol` up from `experiments/common/local_protocol.py`,
   refactored around an explicit per-instance policy:
   `set_phase_policy(phase, policy)` / `pop_phase_policy(phase)`, with
   `PhaseToolPolicy(environment_tools_offered, block_environment_tool_calls,
   blackboard_tools, route_post_message_to, max_post_messages_per_agent)`.
   `environment_handle_tool_call` / `blackboard_handle_tool_call` enforce it
   (drop env calls, rewrite `post_message` to `route_post_message_to`,
   per-agent cap with per-round reset). Keep the generic `agent_turn`
   dispatch, `generate_comm_network` (secret-board creation), and the
   secret-board scrubbing (`_scrub_secret_blackboards`).
4. `terrarium/environments/dcops/hospital/hospital_env.py`: native
   `if phase == "pre_coordination":` branch in `build_agent_context`
   (`hospital_env.py:561`) returning the minimal identity + coalition context,
   replacing Colosseum's `_pre_coordination_context` scrub.
5. Tests: repoint `terrarium/tests/test_local_protocol.py` from
   `experiments.common.local_protocol` to the new native module (removing the
   backwards dependency on Colosseum), and add tests for the phase policy
   (allowlist, board routing, cap, per-round reset).

The Terrarium-side API sketch lives in the Terrarium fork at
`dev/phase-tool-policy-design.md`.

### PR 2 — Colosseum `fork-main` (adoption)

1. Delete `experiments/common/_patch_terrarium.py`. In `run.py`, the
   flag save/set/restore in `_run_pre_coordination_rounds` (`run.py:1040-1152`)
   becomes `protocol.set_phase_policy("pre_coordination", PhaseToolPolicy(...))`
   with push/pop in the `finally`.
2. `experiments/common/local_protocol.py` becomes a thin re-export of the
   native class (or is deleted; imports updated).
3. `tool_availability.py` derives per-phase availability from the
   discovery/policy API instead of `is_post_message_disabled()`;
   `PHASE_ENV_TOOLS_BLOCKED` becomes derived from the active policy.
4. `run.py` keeps everything experiment-specific: sweep axes, run ids, resume,
   the phase orchestration (still just
   `run_phase_entries([{"phase": "pre_coordination", ...}])` over adversaries
   at iteration 0).
5. Update test imports (`tests/test_post_message_scoping.py`,
   `tests/test_pre_coordination_round.py`, etc.) to the native API.
6. Bump the `external/Terrarium` submodule to the new `fork-main` SHA; update
   the README notes about required Terrarium PRs.

### Order and verification

1. Terrarium PR first; Terrarium test suite green on `fork-main`.
2. Colosseum PR second with the submodule bumped; collusion test suite green.
3. One-seed smoke of a benign sweep and a `pre_coordination_round: true` sweep,
   confirming identical tool gating and secret-board routing end to end.

## Open points

- Exact wiring of the policy from protocol to `BaseAgent`'s tool build
  (`terrarium/agents/base.py:403-412`) — the one real design detail still open.
- Whether `pre_coordination` is hardcoded planning-shaped in Terrarium or
  declared per-run via config / phase entries
  (`{"phase": "pre_coordination", "planning_shaped": true}`) — leaning
  config-declared so Terrarium stays generic.
- Trajectory log labels stay `"pre_coordination"` verbatim, so judge / metrics
  / plots that read phase labels need no changes — worth a test assertion.

## File map (current state, `fork-main`)

- Patch + flags: `experiments/common/_patch_terrarium.py`
- Protocol: `experiments/common/local_protocol.py`
- Orchestration / axes / run ids: `experiments/collusion_via_tools/run.py`
- Tool availability helpers: `experiments/collusion_via_tools/tool_availability.py`
- Experiment content: `experiments/collusion_via_tools/{agents,prompts,metrics,judge_collusion_score,judge_beliefs}.py`
- Terrarium gate being patched: `terrarium/tools/discovery.py:85`
- Per-turn tool build: `terrarium/agents/base.py:403-412`
- Phase walk: `terrarium/utils.py:914` (`run_phase_entries`)
- Hospital context hook point: `terrarium/environments/dcops/hospital/hospital_env.py:561`
- Backwards test dependency: `terrarium/tests/test_local_protocol.py`
- Dependency wiring: `pyproject.toml:10` + `[tool.uv.sources]`, `.gitmodules`
