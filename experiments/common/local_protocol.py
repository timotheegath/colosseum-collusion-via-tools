"""Experiment protocol: native SequentialCommunicationProtocol + tool-event recording.

Phase walking (``simulation.phases``), per-phase and per-agent tool gating
(``environment.blocked_tools``), secret-channel creation and enforcement
(``environment.secret_channels``), and the generic ``agent_turn`` are all
native to the Terrarium submodule. This subclass exists only to record every
tool call into ``tool_events`` so the experiments can dump ``tool_events.json``
for metrics/analysis — it performs no gating of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from terrarium.communication_protocols.sequential import SequentialCommunicationProtocol


@dataclass(slots=True)
class ToolEvent:
    tool_name: str
    agent_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    phase: Optional[str]
    iteration: Optional[int]
    planning_round: Optional[int]


class _RunDirToolLogger:
    """Minimal tool_logger stand-in (the protocol only needs ``log_dir``)."""

    def __init__(self, log_dir: Any = None) -> None:
        self.log_dir = log_dir


class LocalCommunicationProtocol(SequentialCommunicationProtocol):
    """In-process protocol backed by the native sequential protocol.

    Everything except tool-event recording is inherited unchanged:

    - ``agent_turn(..., phase=..., round_num=...)`` walks any config-driven phase;
    - ``environment_handle_tool_call`` / ``blackboard_handle_tool_call`` enforce
      ``environment.blocked_tools`` (phase- and agent-scoped) and secret-channel
      ``enforce_in_phases`` rewriting natively;
    - ``generate_comm_network`` creates blackboards (the environment creates the
      ``environment.secret_channels`` boards itself during ``async_init``).
    """

    def __init__(self, *, config: Dict[str, Any], megaboard: Optional[Any] = None) -> None:
        super().__init__(config, tool_logger=_RunDirToolLogger())
        if megaboard is not None:
            self.megaboard = megaboard
        self.tool_events: List[ToolEvent] = []
        self.current_planning_round: Optional[int] = None

    def _record_tool_event(
        self,
        *,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        phase: Optional[str],
        iteration: Optional[int],
    ) -> None:
        self.tool_events.append(
            ToolEvent(
                tool_name=tool_name,
                agent_name=agent_name,
                arguments=dict(arguments or {}),
                result=dict(result or {}),
                phase=phase,
                iteration=iteration,
                planning_round=self.current_planning_round,
            )
        )

    async def agent_turn(
        self,
        agent: Any,
        agent_name: str,
        agent_context: Any,
        environment: Any,
        iteration: int,
        phase: str,
        round_num: Optional[int] = None,
    ) -> Any:
        self.current_planning_round = round_num
        try:
            return await super().agent_turn(
                agent,
                agent_name,
                agent_context,
                environment,
                iteration,
                phase=phase,
                round_num=round_num,
            )
        finally:
            self.current_planning_round = None

    async def environment_handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        *,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = await super().environment_handle_tool_call(
            tool_name, agent_name, arguments, phase=phase, iteration=iteration
        )
        self._record_tool_event(
            tool_name=tool_name,
            agent_name=agent_name,
            arguments=arguments,
            result=result if isinstance(result, dict) else {"result": result},
            phase=phase,
            iteration=iteration,
        )
        return result

    async def blackboard_handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        *,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = await super().blackboard_handle_tool_call(
            tool_name, agent_name, arguments, phase=phase, iteration=iteration
        )
        self._record_tool_event(
            tool_name=tool_name,
            agent_name=agent_name,
            arguments=arguments,
            result=result if isinstance(result, dict) else {"result": result},
            phase=phase,
            iteration=iteration,
        )
        return result
