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


GRAPH_RECURSION_LIMIT = MAX_AGENT_NODE_ITERATIONS * 6 + 16


class AgentGraphState(AgentState, total=False):
    """Private LangGraph state used by the compatibility graph engine."""

    graph_iteration_count: int
    final_answer: str


class AgentGraphEngine(AgentLoopEngine):
    """LangGraph-backed equivalent of the legacy outer Agent loop.

    Graph v0 deliberately keeps the planner, selector and complete ReAct loop
    intact.  Internal routing nodes do not publish public node events, so the
    Runtime-facing event stream remains compatible with AgentLoopEngine.
    """

    def __init__(self) -> None:
        self._compiled_graph = self._build_graph()

    def execute(
        self,
        request: AgentRequest,
        context: AgentRunContext,
        emit: EventEmitter,
    ) -> EngineResult:
        initial_state: AgentGraphState = {
            "question": request.goal,
            "context_memory": context.context_memory,
            "tool_context": ToolContext(
                run_id=context.run_id,
                session_id=context.session_id,
                workspace_root=context.workspace_root,
                granted_permissions=ALL_PERMISSIONS,
            ),
            "_event_callback": emit,
            "graph_iteration_count": 0,
        }
        final_state = self._compiled_graph.invoke(
            initial_state,
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
        return EngineResult(answer=str(final_state.get("final_answer", "")))

    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(AgentGraphState)
        builder.add_node("bootstrap", self._bootstrap)
        builder.add_node("cycle_gate", self._cycle_gate)
        builder.add_node("planner_node", self._planner_graph_node)
        builder.add_node("select_next_step_node", self._select_graph_node)
        builder.add_node("step_started", self._step_started)
        builder.add_node("legacy_agent_loop_node", self._legacy_agent_loop_graph_node)
        builder.add_node("step_finished", self._step_finished)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "bootstrap")
        builder.add_edge("bootstrap", "cycle_gate")
        builder.add_conditional_edges(
            "cycle_gate",
            self._route_cycle,
            {
                "planner_node": "planner_node",
                "select_next_step_node": "select_next_step_node",
            },
        )
        builder.add_edge("planner_node", "cycle_gate")
        builder.add_conditional_edges(
            "select_next_step_node",
            self._route_after_select,
            {
                "finalize": "finalize",
                "step_started": "step_started",
            },
        )
        builder.add_edge("step_started", "legacy_agent_loop_node")
        builder.add_edge("legacy_agent_loop_node", "step_finished")
        builder.add_edge("step_finished", "cycle_gate")
        builder.add_edge("finalize", END)
        return builder.compile()

    @staticmethod
    def _bootstrap(state: AgentGraphState) -> AgentGraphState:
        question = state.get("question", "")
        return {
            **OneRunMemory.initial_state(question=question),
            "graph_iteration_count": 0,
        }

    @staticmethod
    def _cycle_gate(state: AgentGraphState) -> AgentGraphState:
        iteration_count = int(state.get("graph_iteration_count", 0) or 0) + 1
        if iteration_count > MAX_AGENT_NODE_ITERATIONS:
            raise RuntimeError("agent exceeded graph iteration limit")
        return {"graph_iteration_count": iteration_count}

    @staticmethod
    def _route_cycle(state: AgentGraphState) -> str:
        should_plan = (
            state.get("planner_mode") in {"replan", "step_replan"}
            or "plan" not in state
        )
        return "planner_node" if should_plan else "select_next_step_node"

    def _planner_graph_node(self, state: AgentGraphState) -> AgentGraphState:
        emit = self._event_emitter(state)
        merged = self._run_node(
            state=state,
            node_name="planner_node",
            node=planner_node,
            emit=emit,
        )
        self._raise_if_failed(merged)
        return merged

    def _select_graph_node(self, state: AgentGraphState) -> AgentGraphState:
        emit = self._event_emitter(state)
        merged = self._run_node(
            state=state,
            node_name="select_next_step_node",
            node=select_next_step_node,
            emit=emit,
        )
        self._raise_if_failed(merged)
        return merged

    @staticmethod
    def _route_after_select(state: AgentGraphState) -> str:
        if state.get("should_continue_next") == "finish":
            return "finalize"
        return "step_started"

    def _step_started(self, state: AgentGraphState) -> AgentGraphState:
        current_step = _current_step(state)
        if current_step is not None:
            self._event_emitter(state)(
                {
                    "type": "agent.step.started",
                    "step_id": current_step.get("step_id"),
                    "task": current_step.get("task"),
                }
            )
        return {}

    def _legacy_agent_loop_graph_node(
        self,
        state: AgentGraphState,
    ) -> AgentGraphState:
        emit = self._event_emitter(state)
        merged = self._run_node(
            state=state,
            node_name="agent_loop_node",
            node=agent_loop_node,
            emit=emit,
        )
        self._raise_if_failed(merged)
        return merged

    def _step_finished(self, state: AgentGraphState) -> AgentGraphState:
        current_step = _current_step(state)
        if current_step is not None and current_step.get("status") == "done":
            self._event_emitter(state)(
                {
                    "type": "agent.step.finished",
                    "step_id": current_step.get("step_id"),
                    "result": current_step.get("result"),
                }
            )
        return {}

    @staticmethod
    def _finalize(state: AgentGraphState) -> AgentGraphState:
        return {"final_answer": _summarize_agent_answer(state)}

    @staticmethod
    def _event_emitter(state: AgentGraphState) -> EventEmitter:
        emit = state.get("_event_callback")
        if emit is None:
            raise RuntimeError("agent event callback is missing")
        return emit
