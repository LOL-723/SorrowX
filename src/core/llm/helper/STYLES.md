# Agent Runtime Code Style Guide

`AgentRuntime` owns run/session lifecycle, event delivery, trace context, and
failure boundaries. It must not import or mutate `AgentState`.

`AgentEngine` owns task execution. Its public boundary is
`AgentRequest`, `AgentRunContext`, `EngineResult`, and an event callback;
internal node state must not appear in API, CLI, or daemon contracts.

The current `AgentLoopEngine` may use the existing planner and ReAct nodes.
Future Graph engines must implement the same `AgentEngine` protocol so entry
points do not need to change.

Runtime events use stable `type`, `run_id`, `session_id`, and `ts` fields.
Engine events must only provide their event-specific fields; Runtime attaches
the run metadata.
