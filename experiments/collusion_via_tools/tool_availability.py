"""Phase-aware tool availability for the Hospital DCOP prompts.

Terrarium decides which tools an agent can call *per turn* from its
``ToolsetDiscovery`` (``terrarium/agents/base.py``): the environment toolset
for the current phase plus the blackboard toolset. With the submodule's native
gating, ``environment.blocked_tools`` is honoured in both places — phase-wide
via the env tools wrapper, and per agent (``roles``/``agents`` entries) because
``agent_turn`` binds the agent's discovery to the environment and passes the
agent name.

This module exposes the same computation as pure helpers so the prompt layer
can render the exact per-phase toolset without any monkeypatching.

Two phase gates exist natively:

* ``schedule_patient`` is offered by ``HospitalTools.get_tools`` only when
  ``phase == "execution"``.
* ``post_message`` is offered by ``ToolsetDiscovery.get_tools_for_blackboard``
  for every phase except ``"execution"``, unless blocked for the agent in that
  phase via ``environment.blocked_tools``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from terrarium.tools.discovery import ToolsetDiscovery
from terrarium.tools.gates import tool_blocked

# The phases the runner walks the agents through, in prompt-relevant order.
# ``phase_tool_map`` preserves this order so the rendered "it is available in:
# ..." lists are stable.
_PHASES: Tuple[str, ...] = ("planning", "execution", "survey", "pre_coordination")

# Fallback environment toolset, used only when the discovery object yields
# nothing usable. Mirrors ``HospitalTools.get_tools``: four always-on tools
# plus ``schedule_patient`` in the execution phase.
_FALLBACK_ENV_TOOLS_ALWAYS: Tuple[str, ...] = (
    "find_available_slots",
    "get_job_queue",
    "transfer_resources",
    "broadcast_message",
)
_FALLBACK_ENV_TOOLS_EXECUTION_ONLY: Tuple[str, ...] = ("schedule_patient",)

# One shared discovery; bound to the current environment on every call so
# per-agent blocked_tools entries are honoured.
_DISCOVERY = ToolsetDiscovery()


def _extract_names(tools: Any) -> List[str]:
    """Pull ``tool["function"]["name"]`` out of a toolset list, defensively."""
    names: List[str] = []
    for tool in tools or []:
        try:
            name = tool["function"]["name"]
        except (TypeError, KeyError, IndexError):
            continue
        if name:
            names.append(str(name))
    return names


def _env_name(env: Any) -> str:
    if env is None:
        return ""
    return str(
        getattr(env, "tools_environment_name", None) or type(env).__name__ or ""
    )


def _fallback_env_tool_names(phase: str) -> List[str]:
    names = list(_FALLBACK_ENV_TOOLS_ALWAYS)
    if phase == "execution":
        names.extend(_FALLBACK_ENV_TOOLS_EXECUTION_ONLY)
    return names


def _fallback_blackboard_tool_names(phase: str) -> List[str]:
    """Fallback blackboard toolset: post_message outside the execution phase."""
    if phase != "execution":
        return ["post_message"]
    return []


def tool_names_for_phase(agent_name: Optional[str], env: Any, phase: str) -> List[str]:
    """Return the exact tool names ``agent_name`` can call in ``phase``.

    Computed the same way ``terrarium/agents/base.py`` does it: the environment
    toolset (for ``env``) followed by the blackboard toolset, both for ``phase``
    and for this agent, so ``environment.blocked_tools`` entries scoped by
    ``agents``/``roles`` are reflected. Order is preserved (environment tools
    first, then blackboard tools). If the discovery yields nothing usable, this
    falls back to the known Hospital toolset for ``phase``.
    """
    env_name = _env_name(env)
    env_names: List[str] = []
    bb_names: List[str] = []
    if env_name:
        _DISCOVERY.bind_environment(env)
        try:
            env_names = _extract_names(
                _DISCOVERY.get_tools_for_environment(env_name, phase, agent_name=agent_name)
            )
        except Exception:
            env_names = []
        try:
            bb_names = _extract_names(
                _DISCOVERY.get_tools_for_blackboard(phase, env_name, agent_name=agent_name)
            )
        except Exception:
            bb_names = []
    # Fall back per toolset when discovery yields nothing usable for it.
    if not env_names:
        env_names = _fallback_env_tool_names(phase)
    if not bb_names:
        bb_names = _fallback_blackboard_tool_names(phase)
    names = env_names + bb_names
    return [n for n in names if not tool_blocked(env, n, phase, agent_name)]


def phase_tool_map(agent_name: Optional[str], env: Any) -> Dict[str, List[str]]:
    """Map every phase to the tool names offered to ``agent_name`` in it.

    Keys follow ``_PHASES`` order.
    """
    return {phase: tool_names_for_phase(agent_name, env, phase) for phase in _PHASES}
