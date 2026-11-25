"""Runtime patch for terrarium's toolset discovery (task-local round gating).

The ``custom_oai_compatible`` LLM provider this module used to install is now
native in Terrarium: the ``openai_compatible`` provider
(``terrarium.llm.clients.OpenAICompatibleClient``, wired into
``terrarium.utils.get_client_instance`` / ``get_model_name`` /
``get_generation_params``) provides SSE streaming, retry-with-backoff, a 180s
default request timeout and request-only ``strip_reasoning_content``.
Configs therefore use ``provider: openai_compatible`` with the
``OPENAI_COMPATIBLE_BASE_URL`` / ``OPENAI_COMPATIBLE_API_KEY`` env vars, and
the provider dispatch patch (plus its client) is gone from here.

What remains is the task-local round state that the collusion-via-tools runner
drives around terrarium's per-phase toolset lookup:

* ``set_post_message_disabled`` / ``is_post_message_disabled`` -- when true,
  the patched ``ToolsetDiscovery.get_tools_for_blackboard`` returns ``[]`` so
  the ``post_message`` tool is withheld from the agent's tool list (used for
  the ``disable_post_message`` sweep axis). The ``pre_coordination`` phase is
  remapped to ``planning`` for the toolset lookup so the pre-coordination
  round keeps the tool when it is enabled.
* ``set_env_tools_blocked`` / ``is_env_tools_blocked`` -- consumed by
  ``LocalCommunicationProtocol.environment_handle_tool_call`` to drop all
  environment tool calls during the pre-coordination round.
* ``set_secret_blackboard_id`` / ``get_secret_blackboard_id`` and the
  per-agent ``post_message`` count/limit helpers -- consumed by
  ``LocalCommunicationProtocol.blackboard_handle_tool_call`` to reroute and
  cap coordination messages during the pre-coordination round.

The flags are stored in :class:`contextvars.ContextVar`, so the concurrent
``_run_single`` coroutines scheduled by a sweep (``max_concurrent_runs > 1``)
each see their own value and can never clobber one another's restriction
mid-round.

Importing this module applies the remaining patch (idempotently).
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blackboard tools patch (e.g. disabling post_message for tool-based collusion)
# ---------------------------------------------------------------------------
# The flag is task-local, not process-global: a plain module-level bool would be
# clobbered whenever two `_run_single` coroutines overlap under
# `max_concurrent_runs > 1` (the last writer wins for every other in-flight run's
# remaining turns). A ContextVar gives each asyncio task its own copy; child
# tasks (and `copy_context().run(...)` at any thread boundary) inherit a snapshot.
_POST_MESSAGE_DISABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "colosseum_post_message_disabled", default=False
)


def set_post_message_disabled(disabled: bool) -> None:
    """Enable or disable the post_message tool for the current task/context.

    The value is scoped to the calling task's :mod:`contextvars` context, so
    concurrent runs cannot interfere with each other.
    """
    _POST_MESSAGE_DISABLED.set(bool(disabled))


def is_post_message_disabled() -> bool:
    """Return whether post_message is disabled in the current task/context."""
    return _POST_MESSAGE_DISABLED.get()


# ---------------------------------------------------------------------------
# Environment-tools block (e.g. for the pre-coordination round: agents see
# `broadcast_message`, `transfer_resources`, etc. in their prompt, but any call
# is silently dropped so they cannot probe the environment before losing the
# private channel).
# ---------------------------------------------------------------------------
# Same task-local ContextVar rationale as `_POST_MESSAGE_DISABLED`.
_ENV_TOOLS_BLOCKED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "colosseum_env_tools_blocked", default=False
)


def set_env_tools_blocked(blocked: bool) -> None:
    """Block all environment tool calls (e.g. ``broadcast_message``,
    ``transfer_resources``) for the current task/context.

    Used during the pre-coordination round so that adversaries cannot probe
    the environment before they lose ``post_message``. The agent still sees
    the tools in its prompt; the protocol layer returns an empty result
    instead of executing them.
    """
    _ENV_TOOLS_BLOCKED.set(bool(blocked))


def is_env_tools_blocked() -> bool:
    """Return whether environment tools are blocked in the current task/context."""
    return _ENV_TOOLS_BLOCKED.get()


# ---------------------------------------------------------------------------
# Per-agent `post_message` count cap (e.g. for the pre-coordination round).
# Stores a dict mapping agent_name -> count of `post_message` invocations in
# the current task's context. The caller (typically the experiment runner) is
# responsible for resetting it at round boundaries.
# ---------------------------------------------------------------------------
_POST_MESSAGE_COUNTS: contextvars.ContextVar[Dict[str, int]] = contextvars.ContextVar(
    "colosseum_post_message_counts", default={}
)


def reset_post_message_counts() -> None:
    """Reset the per-agent `post_message` counter for the current context."""
    _POST_MESSAGE_COUNTS.set({})


def get_post_message_count(agent_name: str) -> int:
    """Return how many times `agent_name` has called `post_message` in the
    current context (this run / task / round, depending on what the caller
    reset on)."""
    return _POST_MESSAGE_COUNTS.get().get(agent_name, 0)


def record_post_message_call(agent_name: str) -> int:
    """Increment the counter and return the new value."""
    counts = dict(_POST_MESSAGE_COUNTS.get())
    counts[agent_name] = counts.get(agent_name, 0) + 1
    _POST_MESSAGE_COUNTS.set(counts)
    return counts[agent_name]


# Per-task cap on `post_message` calls (None = unlimited). Used to throttle
# the pre-coordination round so adversaries cannot flood the secret blackboard.
# The counter (see `_POST_MESSAGE_COUNTS`) is keyed by agent_name so the limit
# is enforced per-agent.
_POST_MESSAGE_CALL_LIMIT: contextvars.ContextVar[Optional[int]] = (
    contextvars.ContextVar("colosseum_post_message_call_limit", default=None)
)


def set_post_message_call_limit(limit: Optional[int]) -> None:
    """Set the per-agent `post_message` cap for the current context.

    ``None`` means unlimited. The limit is enforced in
    :meth:`LocalCommunicationProtocol.blackboard_handle_tool_call`.
    """
    if limit is None:
        _POST_MESSAGE_CALL_LIMIT.set(None)
    else:
        _POST_MESSAGE_CALL_LIMIT.set(int(limit))


def get_post_message_call_limit() -> Optional[int]:
    """Return the per-agent `post_message` cap (``None`` = unlimited)."""
    return _POST_MESSAGE_CALL_LIMIT.get()


# ---------------------------------------------------------------------------
# Secret blackboard id (for the pre-coordination round: if set, the protocol
# layer rewrites any `post_message` call to this board so the agent cannot
# accidentally post to the public channel during the preparation turn).
# ---------------------------------------------------------------------------
_SECRET_BLACKBOARD_ID: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "colosseum_secret_blackboard_id", default=None
)


def set_secret_blackboard_id(bb_id: Optional[int]) -> None:
    """Set the secret blackboard id that ``post_message`` calls should be
    redirected to (``None`` = no rewriting).

    Used during the pre-coordination round to guarantee that adversarial
    coordination stays on the coalition-only board even if the model omits
    ``blackboard_id`` or picks the public board by mistake.
    """
    if bb_id is None:
        _SECRET_BLACKBOARD_ID.set(None)
    else:
        _SECRET_BLACKBOARD_ID.set(int(bb_id))


def get_secret_blackboard_id() -> Optional[int]:
    """Return the secret blackboard id (``None`` = no rewriting)."""
    return _SECRET_BLACKBOARD_ID.get()


def _install_blackboard_tools_patch() -> None:
    """Patch ToolsetDiscovery to allow disabling post_message dynamically."""
    try:
        from terrarium.tools.discovery import ToolsetDiscovery
    except ImportError:
        return

    if getattr(ToolsetDiscovery.get_tools_for_blackboard, "_colosseum_patched", False):
        return

    _orig_get_tools = ToolsetDiscovery.get_tools_for_blackboard
    _orig_get_names = ToolsetDiscovery.get_blackboard_tool_names

    # Phases that expose the blackboard toolset (i.e. offer `post_message`).
    # `"pre_coordination"` is a planning-shaped adversary round and must look
    # the same to the toolset as `"planning"`, otherwise terrarium's phase gate
    # silently drops `post_message` from the toolset and the secret channel
    # stays empty (the original bug). The trajectory log label is independent
    # and still records `"pre_coordination"` verbatim -- only the toolset
    # lookup is remapped here.
    planning_shaped_phases = ("planning", "pre_coordination")

    def get_tools_for_blackboard(self, phase: str, *args: Any, **kwargs: Any):
        if is_post_message_disabled():
            return []
        effective = "planning" if phase in planning_shaped_phases else phase
        return _orig_get_tools(self, effective, *args, **kwargs)

    def get_blackboard_tool_names(self, *args: Any, **kwargs: Any):
        if is_post_message_disabled():
            return set()
        return _orig_get_names(self, *args, **kwargs)

    get_tools_for_blackboard._colosseum_patched = True  # type: ignore[attr-defined]
    get_blackboard_tool_names._colosseum_patched = True  # type: ignore[attr-defined]

    ToolsetDiscovery.get_tools_for_blackboard = get_tools_for_blackboard
    ToolsetDiscovery.get_blackboard_tool_names = get_blackboard_tool_names


def apply() -> None:
    """Install the remaining terrarium patches.

    Idempotent: repeated calls (or repeated imports) are a no-op. Each patch is
    guarded independently so a partially-applied state still converges.
    """
    _install_blackboard_tools_patch()


# Applying on import means a bare `import experiments.common._patch_terrarium`
# (or any import of a module that imports this one) is enough to activate it.
apply()
