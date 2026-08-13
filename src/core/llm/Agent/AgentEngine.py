import json
from typing import Any, Callable, Protocol

from llm.Agent.AgentRuntime import AgentRequest, AgentRunContext, EngineResult
from llm.Agent.memory import OneRunMemory
from llm.Agent.nodes import agent_loop_node, planner_node, select_next_step_node
from llm.Agent.nodes.universal import _chat_completion
from llm.Agent.tools.contracts import ALL_PERMISSIONS, ToolContext
from llm.Agent.prompt import FINAL_RESULT_SUMMARY_PROMPT
from llm.Agent.state import (
    AgentEventCallback,
    AgentState,
    MAX_PLAN_STEPS,
    MAX_REPLAN_COUNT,
    MAX_STEP_REPLAN_COUNT,
    PlanStepState,
)


MAX_AGENT_NODE_ITERATIONS = MAX_PLAN_STEPS * (
    1 + MAX_REPLAN_COUNT + MAX_STEP_REPLAN_COUNT
) + 3
EventEmitter = AgentEventCallback


class AgentEngine(Protocol):
    def execute(
        self,
        request: AgentRequest,
        context: AgentRunContext,
        emit: EventEmitter,
    ) -> EngineResult:
        ...


class AgentLoopEngine:
    def execute(
        self,
        request: AgentRequest,
        context: AgentRunContext,
        emit: EventEmitter,
    ) -> EngineResult:
        agent_state = OneRunMemory.initial_state(question=request.goal)
        agent_state["_event_callback"] = emit
        agent_state["context_memory"] = context.context_memory
        agent_state["tool_context"] = ToolContext(
            run_id=context.run_id,
            session_id=context.session_id,
            workspace_root=context.workspace_root,
            granted_permissions=ALL_PERMISSIONS,
        )

        for _ in range(MAX_AGENT_NODE_ITERATIONS):
            should_plan = (
                agent_state.get("planner_mode") in {"replan", "step_replan"}
                or "plan" not in agent_state
            )
            if should_plan:
                agent_state = self._run_node(
                    state=agent_state,
                    node_name="planner_node",
                    node=planner_node,
                    emit=emit,
                )
                self._raise_if_failed(agent_state)
                continue

            agent_state = self._run_node(
                state=agent_state,
                node_name="select_next_step_node",
                node=select_next_step_node,
                emit=emit,
            )
            self._raise_if_failed(agent_state)
            if agent_state.get("should_continue_next") == "finish":
                answer = _summarize_agent_answer(agent_state)
                return EngineResult(answer=answer)

            current_step = _current_step(agent_state)
            if current_step is not None:
                emit(
                    {
                        "type": "agent.step.started",
                        "step_id": current_step.get("step_id"),
                        "task": current_step.get("task"),
                    }
                )

            agent_state = self._run_node(
                state=agent_state,
                node_name="agent_loop_node",
                node=agent_loop_node,
                emit=emit,
            )
            self._raise_if_failed(agent_state)

            finished_step = _current_step(agent_state)
            if finished_step is not None and finished_step.get("status") == "done":
                emit(
                    {
                        "type": "agent.step.finished",
                        "step_id": finished_step.get("step_id"),
                        "result": finished_step.get("result"),
                    }
                )

        raise RuntimeError("agent exceeded graph iteration limit")

    def _run_node(
        self,
        *,
        state: AgentState,
        node_name: str,
        node: Callable[[AgentState], AgentState],
        emit: EventEmitter,
    ) -> AgentState:
        previous_log_count = len(state.get("logs", []))
        emit({"type": "agent.node.started", "node": node_name})
        update = node(state)
        merged = _merge_agent_state(state, update)
        for log_item in merged.get("logs", [])[previous_log_count:]:
            emit(
                {
                    "type": "agent.log",
                    "source": log_item.get("node", node_name),
                    "message": log_item.get("message", ""),
                    "extra": {
                        key: value
                        for key, value in log_item.items()
                        if key not in {"node", "message"}
                    },
                }
            )
        emit(
            {
                "type": "agent.node.finished",
                "node": node_name,
                "status": merged.get("agent_status", "running"),
                "phase": merged.get("phase"),
            }
        )
        return merged

    def _raise_if_failed(self, agent_state: AgentState) -> None:
        if agent_state.get("agent_status") == "failed":
            raise RuntimeError(str(agent_state.get("error") or "agent failed"))


def _merge_agent_state(state: AgentState, update: AgentState) -> AgentState:
    merged: AgentState = {**state, **update}
    return merged


def _current_step(agent_state: AgentState) -> PlanStepState | None:
    current_step_id = agent_state.get("current_step_id")
    if not current_step_id:
        return None
    for step in agent_state.get("plan", []):
        if step.get("step_id") == current_step_id:
            return step
    return None


def _summarize_agent_answer(agent_state: AgentState) -> str:
    payload = {
        "question": agent_state.get("question", ""),
        "plan": agent_state.get("plan", []),
        "step_results": agent_state.get("step_results", []),
        "failed_tools": agent_state.get("failed_tools", []),
    }
    content = _chat_completion(
        user_message=json.dumps(payload, ensure_ascii=False),
        system_prompt=FINAL_RESULT_SUMMARY_PROMPT,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(data, dict):
        return content
    final_answer = data.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer
    return content
